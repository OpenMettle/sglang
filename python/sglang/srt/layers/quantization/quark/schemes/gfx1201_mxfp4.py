# SPDX-License-Identifier: Apache-2.0
"""Dense MXFP4 activation quantizer and FP4 GEMM for gfx1201 without AITER.

gfx1201 (RDNA4, Radeon AI PRO R9700) has no AITER build. The two Triton
kernels in gfx1201_mxfp4_kernels.py are AMD's AITER kernels (MIT), vendored
from ROCm/aiter 32214c295ea2eec1055bdb3467d2eee7289257be by the qualified
gfx1201 serving runtime; that file is byte-identical to the qualified copy.

The launch geometry is the qualified runtime's. The GEMM keeps BLOCK_SIZE_K
and NUM_KSPLIT fixed, so the K accumulation order is the same for every row
count; only the M/N tile and warp count follow the row count.

Two verify-row (M8/M16, the DFLASH 8-token block at batch 1-2) launch changes
from the op234 runtime are available, each off unless its variable is 1:

- SGLANG_GFX1201_GATE_UP_WAVE0: waves_per_eu=0 for the gate/up shard
  (N=17408, K=5120); op220 measured about 40 us less per GEMM.
- SGLANG_GFX1201_SPLITK_DOWNOUT: four K partitions for the down (5120x8704)
  and GDN/attention output (5120x3072) shards, FP32 partials reduced in a
  fixed order ((p0 + p1) + p2) + p3. Deterministic, no atomics; not bitwise
  equal to the single partition (op234 S4).
- SGLANG_GFX1201_SPLITK_WIDE: the same split-K for the remaining under-filled
  verify shards: four partitions for the packed GDN input projection
  (8240x5120) and attention QKV (7168x5120), two for gate/up (17408x5120).
  A graph-captured, DRAM-fed sweep (2026-11-09) measured 80.0 to 66.8,
  64.4 to 58.1 and 138.0 to 132.7 us per call at M8, with outputs bitwise
  equal to the single partition on random data (not guaranteed in general).
- SGLANG_GFX1201_SPLITK_FUSED_REDUCE: run those split-K GEMMs as one kernel,
  the last partition of each tile summing the partials in the same fixed order
  (gfx1201_mxfp4_splitk), instead of a GEMM plus a reduce kernel. Bitwise equal
  to the two-kernel path. Each compiled variant's code is checked once
  (fused_splitk_asm_ok); a variant that fails the check uses the two-kernel path.
"""

import logging
import os
import re

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.quark.schemes import gfx1201_mxfp4_kernels as _k


logger = logging.getLogger(__name__)
_splitk_routed = set()

def _enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes")


GATE_UP_WAVE0 = _enabled("SGLANG_GFX1201_GATE_UP_WAVE0")
SPLITK_DOWNOUT = _enabled("SGLANG_GFX1201_SPLITK_DOWNOUT")
SPLITK_WIDE = _enabled("SGLANG_GFX1201_SPLITK_WIDE")
SPLITK_FUSED_REDUCE = _enabled("SGLANG_GFX1201_SPLITK_FUSED_REDUCE")
_VERIFY_ROWS = (8, 16)
_GATE_UP_SHAPE = (17408, 5120)
# (N, K) -> K partitions; every partition covers whole BLOCK_SIZE_K=128 blocks.
_SPLITK_DOWNOUT_PARTS = {(5120, 8704): 4, (5120, 3072): 4}
_SPLITK_WIDE_PARTS = {(8240, 5120): 4, (7168, 5120): 4, (17408, 5120): 2}
_SPLITK_REDUCE_BLOCK = 1024
# One FP32 partial buffer per device, sized for the largest split launch
# (2 x 16 x 17408 floats = 2.1 MiB). GEMMs on a stream run in order, so every
# layer shares it; it is allocated outside graph capture and reused by every replay.
_SPLITK_SCRATCH_FLOATS = max(parts * max(_VERIFY_ROWS) * n
                             for table in (_SPLITK_DOWNOUT_PARTS, _SPLITK_WIDE_PARTS)
                             for (n, _), parts in table.items())
_splitk_scratch: dict = {}
# One int32 arrival counter per output tile for the fused reduce, shared by every
# launch on the device (each launch leaves its counters at zero). The largest
# verify grid has cdiv(16, 16) * cdiv(17408, 64) = 272 tiles.
_SPLITK_COUNTERS = 4096
_splitk_counters: dict = {}
_fused_asm_checked: dict = {}

_GEMM_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "NUM_KSPLIT": 1,
    "cache_modifier": None,
    "matrix_instr_nonkdim": 16,
    "num_stages": 1,
    "num_warps": 4,
    "waves_per_eu": 1,
}


def _quant_geometry(m: int, k: int) -> dict:
    if m <= 32:
        block_m, block_n, iterations, warps, stages = 1 << (m - 1).bit_length(), 32, 1, 1, 1
    else:
        block_m, block_n, iterations, warps, stages = 32, 128, 4, 4, 2
    return dict(
        grid=(triton.cdiv(m, block_m), triton.cdiv(k, block_n * iterations)),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        NUM_ITER=iterations,
        NUM_STAGES=stages,
        num_warps=warps,
        waves_per_eu=0,
        num_stages=1,
    )


def dynamic_mxfp4_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 rows to packed FP4 (M, K/2) and E8M0 scales (M, K/32)."""
    if x.dim() != 2 or x.dtype != torch.bfloat16:
        raise ValueError("gfx1201 MXFP4 quantizer takes 2-D BF16 activations")
    x = x.contiguous()
    m, k = x.shape
    if m == 0 or k % 32:
        raise ValueError(f"gfx1201 MXFP4 quantizer needs M > 0 and K % 32 == 0, got {m}x{k}")
    packed = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    # Same transposed scale allocation as AITER's dynamic_mxfp4_quant.
    scales = torch.empty_strided((m, k // 32), (1, m), dtype=torch.uint8, device=x.device)
    geometry = _quant_geometry(m, k)
    grid = geometry.pop("grid")
    _k._dynamic_mxfp4_quant_kernel[grid](
        x, packed, scales, *x.stride(), *packed.stride(), *scales.stride(),
        M=m, N=k, MXFP4_QUANT_BLOCK_SIZE=32, SCALING_MODE=0, **geometry,
    )
    return packed, scales


@triton.jit
def _splitk_reduce_kernel(p_ptr, c_ptr, MN, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    # Contiguous FP32 partials (PARTS, M, N), MN = M * N. Fixed order
    # ((p0 + p1) + p2) + ..., one rounding.
    offs = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.load(p_ptr + offs, mask=mask, other=0.0)
    for i in tl.static_range(1, PARTS):
        acc = acc + tl.load(p_ptr + i * MN + offs, mask=mask, other=0.0)
    tl.store(c_ptr + offs, acc.to(c_ptr.type.element_ty), mask=mask)


def _splitk_partials(device: torch.device, m: int, n: int, parts: int):
    """The shared FP32 partial buffer viewed as (parts, M, N), or None during a
    graph capture that has not seen it allocated (the caller then launches
    the single partition, so capture never allocates from the graph pool)."""
    scratch = _splitk_scratch.get(device.index)
    if scratch is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        scratch = torch.empty(_SPLITK_SCRATCH_FLOATS, dtype=torch.float32, device=device)
        _splitk_scratch[device.index] = scratch
        _splitk_counters[device.index] = torch.zeros(_SPLITK_COUNTERS, dtype=torch.int32, device=device)
    return scratch[: parts * m * n].view(parts, m, n)


def fused_splitk_asm_ok(asm: str):
    """Check the compiled fused split-K kernel keeps the ordering it relies on.

    Returns (ok, reason). Required, in the code before the counter atomic: an
    s_wait_storecnt 0x0 between the last partial store and the workgroup barrier
    (so every wave's stores are complete before the count is published); the
    atomic at device scope; and, after it, device- or system-scope partial loads.
    """
    lines = [l.strip() for l in asm.splitlines() if l.strip() and not l.strip().startswith((";", "."))]
    atom = [i for i, l in enumerate(lines) if re.match(r"(global|buffer)_atomic_add_u32", l)]
    if len(atom) != 1:
        return False, f"expected one counter atomic, found {len(atom)}"
    a = atom[0]
    if "scope:SCOPE_DEV" not in lines[a] and "scope:SCOPE_SYS" not in lines[a]:
        return False, "counter atomic is not device scope"
    stores = [i for i in range(a) if re.match(r"(global|buffer)_store", lines[i])]
    bar = [i for i in range(a) if lines[i].startswith("s_barrier_signal")]
    if not stores or not bar:
        return False, "no partial store or barrier before the atomic"
    last_store = stores[-1]
    first_bar = next((i for i in bar if i > last_store), None)
    if first_bar is None:
        return False, "no barrier after the last partial store"
    if not any(lines[i].startswith(("s_wait_storecnt 0x0", "s_wait_storecnt_dscnt 0x0")) for i in range(last_store + 1, first_bar)):
        return False, "partial stores are not drained (s_wait_storecnt 0x0) before the barrier"
    loads = [l for l in lines[a + 1:] if re.match(r"(global|buffer)_load", l)]
    if not loads or not all(("scope:SCOPE_DEV" in l or "scope:SCOPE_SYS" in l) for l in loads):
        return False, "partial reloads are not device scope"
    return True, ""


def _launch_fused(x, w, x_scales, w_scales, partials, y, counters, m, n, packed_k, config, grid):
    from sglang.srt.layers.quantization.quark.schemes import gfx1201_mxfp4_splitk as _fk

    return _fk._gemm_afp4wfp4_splitk_fused_kernel[grid](
        x, w, partials, y, counters, x_scales, w_scales,
        m, n, packed_k,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),
        partials.stride(0), partials.stride(1), partials.stride(2),
        y.stride(0), y.stride(1),
        x_scales.stride(0), x_scales.stride(1),
        w_scales.stride(0), w_scales.stride(1),
        **config,
    )


def _launch(x, w, x_scales, w_scales, c, m, n, packed_k, config, grid, c_strides):
    _k._gemm_afp4wfp4_kernel[grid](
        x, w, c, x_scales, w_scales,
        m, n, packed_k,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),
        *c_strides,
        x_scales.stride(0), x_scales.stride(1),
        w_scales.stride(0), w_scales.stride(1),
        **config,
    )


def _splitk_parts(n: int, k: int, splitk_downout, splitk_wide) -> int:
    """K partitions for a verify-row GEMM of this shard shape (1 = no split)."""
    if (n, k) in _SPLITK_DOWNOUT_PARTS and (SPLITK_DOWNOUT if splitk_downout is None else splitk_downout):
        return _SPLITK_DOWNOUT_PARTS[(n, k)]
    if (n, k) in _SPLITK_WIDE_PARTS and (SPLITK_WIDE if splitk_wide is None else splitk_wide):
        return _SPLITK_WIDE_PARTS[(n, k)]
    return 1


def gemm_afp4wfp4(x, w, x_scales, w_scales, dtype, y, *, gate_up_wave0=None, splitk_downout=None,
                  splitk_wide=None):
    """y[M, N] = dequant(x) @ dequant(w).T for packed FP4 x (M, K/2) and w (N, K/2).

    gate_up_wave0 / splitk_downout / splitk_wide override the environment
    defaults (tests).
    """
    m, packed_k = x.shape
    n, w_packed_k = w.shape
    if w_packed_k != packed_k:
        raise ValueError("packed reduction dimension mismatch")
    if tuple(y.shape) != (m, n):
        raise ValueError("output shape mismatch")
    k = packed_k * 2
    block_m, block_n, warps = (16, 64, 4) if m <= 16 else (128, 128, 8) if m >= 1024 else (64, 64, 4)
    config = dict(_GEMM_CONFIG, BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, num_warps=warps, SPLITK_BLOCK_SIZE=k)
    verify_rows = m in _VERIFY_ROWS
    if verify_rows and (n, k) == _GATE_UP_SHAPE and (GATE_UP_WAVE0 if gate_up_wave0 is None else gate_up_wave0):
        config["waves_per_eu"] = 0
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    parts = _splitk_parts(n, k, splitk_downout, splitk_wide) if verify_rows else 1
    if parts > 1:
        # Each partition covers K/parts columns (2176 down, 768 out, 1280 or
        # 2560 for K=5120): whole BLOCK_SIZE_K=128 blocks, so every partition is EVEN_K.
        partials = _splitk_partials(y.device, m, n, parts)
        if partials is not None:
            if (m, n, k) not in _splitk_routed:
                _splitk_routed.add((m, n, k))
                logger.info("Split-K MXFP4 verify GEMM (gfx1201) route: %d rows, N=%d K=%d, %d partitions",
                            m, n, k, parts)
            split = dict(config, NUM_KSPLIT=parts, SPLITK_BLOCK_SIZE=k // parts)
            key = (y.device.index, m, n, k, parts)
            if SPLITK_FUSED_REDUCE and _fused_asm_checked.get(key, True):
                if k % (parts * config["BLOCK_SIZE_K"]) or grid[0] > _SPLITK_COUNTERS:
                    raise ValueError("fused split-K needs whole K blocks per partition and a counter per tile")
                compiled = _launch_fused(x, w, x_scales, w_scales, partials, y, _splitk_counters[y.device.index],
                                         m, n, packed_k, split, (grid[0] * parts,))
                if key not in _fused_asm_checked:
                    ok, reason = fused_splitk_asm_ok(compiled.asm["amdgcn"])
                    _fused_asm_checked[key] = ok
                    if ok:
                        logger.info("Fused split-K reduce (gfx1201) route: %d rows, N=%d K=%d, %d partitions",
                                    m, n, k, parts)
                    else:
                        # This launch already ran: recompute it on the checked two-kernel path.
                        logger.error("Fused split-K reduce disabled for %d rows, N=%d K=%d: %s", m, n, k, reason)
                if _fused_asm_checked[key]:
                    return y
            _launch(x, w, x_scales, w_scales, partials, m, n, packed_k, split, (grid[0] * parts,),
                    (partials.stride(0), partials.stride(1), partials.stride(2)))
            _splitk_reduce_kernel[(triton.cdiv(m * n, _SPLITK_REDUCE_BLOCK),)](
                partials, y, m * n, PARTS=parts, BLOCK=_SPLITK_REDUCE_BLOCK, num_warps=4)
            return y
    _launch(x, w, x_scales, w_scales, y, m, n, packed_k, config, grid, (0, y.stride(0), y.stride(1)))
    return y
