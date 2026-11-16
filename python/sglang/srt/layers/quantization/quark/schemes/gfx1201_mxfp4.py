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
"""

import logging
import os

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
_VERIFY_ROWS = (8, 16)
_GATE_UP_SHAPE = (17408, 5120)
_SPLITK_SHAPES = ((5120, 8704), (5120, 3072))
_SPLITK_PARTS = 4
_SPLITK_REDUCE_BLOCK = 1024
# One FP32 partial buffer per device, sized for the largest split launch
# (4 x 16 x 5120 = 1.25 MiB). GEMMs on a stream run in order, so every layer
# shares it; it is allocated outside graph capture and reused by every replay.
_splitk_scratch: dict = {}

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
def _splitk_reduce_kernel(p_ptr, c_ptr, MN, BLOCK: tl.constexpr):
    # Contiguous FP32 partials (4, M, N), MN = M * N. Fixed order, one rounding.
    offs = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.load(p_ptr + offs, mask=mask, other=0.0)
    acc = acc + tl.load(p_ptr + MN + offs, mask=mask, other=0.0)
    acc = acc + tl.load(p_ptr + 2 * MN + offs, mask=mask, other=0.0)
    acc = acc + tl.load(p_ptr + 3 * MN + offs, mask=mask, other=0.0)
    tl.store(c_ptr + offs, acc.to(c_ptr.type.element_ty), mask=mask)


def _splitk_partials(device: torch.device, m: int, n: int):
    """The shared FP32 partial buffer viewed as (4, M, N), or None during a
    graph capture that has not seen it allocated (the caller then launches
    the single partition, so capture never allocates from the graph pool)."""
    scratch = _splitk_scratch.get(device.index)
    if scratch is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        scratch = torch.empty(_SPLITK_PARTS * 16 * 5120, dtype=torch.float32, device=device)
        _splitk_scratch[device.index] = scratch
    return scratch[: _SPLITK_PARTS * m * n].view(_SPLITK_PARTS, m, n)


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


def gemm_afp4wfp4(x, w, x_scales, w_scales, dtype, y, *, gate_up_wave0=None, splitk_downout=None):
    """y[M, N] = dequant(x) @ dequant(w).T for packed FP4 x (M, K/2) and w (N, K/2).

    gate_up_wave0 / splitk_downout override the environment defaults (tests).
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
    if verify_rows and (n, k) in _SPLITK_SHAPES and (SPLITK_DOWNOUT if splitk_downout is None else splitk_downout):
        # Each partition covers K/4 columns (2176 down, 768 out): whole
        # BLOCK_SIZE_K=128 blocks, so every partition is EVEN_K.
        partials = _splitk_partials(y.device, m, n)
        if partials is not None:
            if (m, n, k) not in _splitk_routed:
                _splitk_routed.add((m, n, k))
                logger.info("Split-K MXFP4 verify GEMM (gfx1201) route: %d rows, N=%d K=%d", m, n, k)
            split = dict(config, NUM_KSPLIT=_SPLITK_PARTS, SPLITK_BLOCK_SIZE=k // _SPLITK_PARTS)
            _launch(x, w, x_scales, w_scales, partials, m, n, packed_k, split, (grid[0] * _SPLITK_PARTS,),
                    (partials.stride(0), partials.stride(1), partials.stride(2)))
            _splitk_reduce_kernel[(triton.cdiv(m * n, _SPLITK_REDUCE_BLOCK),)](
                partials, y, m * n, BLOCK=_SPLITK_REDUCE_BLOCK, num_warps=4)
            return y
    _launch(x, w, x_scales, w_scales, y, m, n, packed_k, config, grid, (0, y.stride(0), y.stride(1)))
    return y
