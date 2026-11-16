# SPDX-License-Identifier: Apache-2.0
"""Speculative target-verify attention (a causal chain of ``T <= 8`` draft tokens per
request) for the Triton attention backend on gfx115x (RDNA3.5; tuned on gfx1151).

``extend_attention_fwd`` walks the cached prefix once per query block, and the split-KV
verify kernels are gfx95-only. Here one program per (request, kv head, group of ``HG`` q
heads, split) reads its share of the prefix once for all ``T x HG`` query rows of the
request and runs QK and PV on the matrix cores; a second kernel merges the splits. Row
``t`` sees ``seq_lens[b] + t + 1`` tokens: the committed prefix plus block tokens
``0..t``, whose K/V the backend has already written to the cache.

Numerics follow the extend kernel: fp32 online softmax, ``p`` rounded to bf16 for PV. On
gfx1151 with Qwen3.8 shapes (24 q heads, 4 kv heads, head dim 256) this takes 0.10 / 0.24 /
1.0 ms per layer at 1k / 8k / 39k prefixes against 0.43 / 3.5 / 16 ms for
``extend_attention_fwd``.
"""

import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention import gfx1201_fp8_kv
from sglang.srt.utils import get_bool_env_var, is_hip, use_gfx115_kernels

logger = logging.getLogger(__name__)

# (BT, warps, HG) = (32, 4, 6) and 20 splits per kv head, swept on gfx1151 with Qwen3.8
# shapes; HG = 3 avoids the accumulator's register spills but is 2-3x slower
NSPLIT = 20
# gfx1201 (R9700, 64 CUs), Qwen3.8 at TP2 (12 q heads, 2 kv heads): 32 splits is the best
# single setting for one and two requests from 1k to 88k prefixes (20 to 96 swept)
NSPLIT_GFX1201 = 32
# SGLANG_GFX1201_VERIFY_ATTENTION_B1_TILE=1: one request per batch runs (BT, warps, QC) =
# (64, 8, 64) at 32 splits. A graph-captured, DRAM-fed sweep on gfx1201 (16 layers, each on
# its own KV cache) measured 96.6 / 178 / 321 / 434 against 114.5 / 219 / 406 / 560 us per
# layer at 23.5k / 47k / 88k / 120k prefixes (+1.3 us at 1k); at two requests it is slower
# than the default tile, so batches of two or more keep it.
GFX1201_B1_TILE = (64, 8, 64)
# With an fp8 cache under gfx1201_fp8_kv.DECODE_PAIRS, batches of one or two requests run
# (64, 4, 64). Measured the same way against the bf16 cache on its own tile: 54.6 / 111 /
# 207 / 369 / 506 against 56.6 / 113 / 208 / 367 / 500 us per layer at 8k / 23.5k / 47k /
# 88k / 123k for one request, 312 against 367 us at 47k for two; with 8 warps the fp8 cache
# took 246 / 440 us at 47k / 88k.
GFX1201_FP8_B1_TILE = (64, 4, 64)
_routed = set()


@lru_cache(maxsize=1)
def gfx1201_enabled() -> bool:
    """SGLANG_GFX1201_VERIFY_ATTENTION=1 on gfx1201: run target verify through this kernel."""
    if not (is_hip() and get_bool_env_var("SGLANG_GFX1201_VERIFY_ATTENTION")):
        return False
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1201"
    except Exception:
        return False


@lru_cache(maxsize=1)
def gfx1201_b1_tile_enabled() -> bool:
    """SGLANG_GFX1201_VERIFY_ATTENTION_B1_TILE=1 with the gfx1201 route on."""
    return gfx1201_enabled() and get_bool_env_var("SGLANG_GFX1201_VERIFY_ATTENTION_B1_TILE")


def enabled() -> bool:
    """The kernel serves target verify on this device."""
    return use_gfx115_kernels() or gfx1201_enabled()


@triton.jit
def _verify_split_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    part_ptr,
    ml_ptr,
    req_to_token,
    r2t_stride,
    req_pool_indices,
    seq_lens,
    scale,
    T: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    NSPLIT: tl.constexpr,
    BT: tl.constexpr,
    HG: tl.constexpr,
    QC: tl.constexpr,
    DECODE: tl.constexpr = 0,
):
    """One program per (request, kv head, group of ``HG`` of its q heads, split); rows are
    (block token, q head of the group), 8 x next_pow2(HG). The groups of a kv head are
    neighbouring programs, so the second read of a token range mostly hits the cache."""
    GQ: tl.constexpr = HQ // HKV
    NG: tl.constexpr = (GQ + HG - 1) // HG
    HP: tl.constexpr = 4 if HG <= 4 else 8
    R: tl.constexpr = 8 * HP
    u = tl.program_id(0)
    g = u % NG
    j = (u // NG) % HKV
    s = (u // (NG * HKV)) % NSPLIT
    b = u // (NG * HKV * NSPLIT)
    prefix = tl.load(seq_lens + b).to(tl.int32)
    rp = tl.load(req_pool_indices + b).to(tl.int64)
    total = prefix + T
    chunk = tl.cdiv(total, NSPLIT)
    t0 = s * chunk
    t1 = tl.minimum(total, t0 + chunk)
    r = tl.arange(0, R)
    m = r // HP
    h = g * HG + r % HP
    valid = (r % HP < HG) & (h < GQ) & (m < T)
    seq_r = prefix + m + 1
    d = tl.arange(0, D)
    dq = tl.arange(0, QC)
    qrow = (b * T + m) * HQ + j * GQ + h
    # padding rows read a real q row (finite scores) and are never stored
    q_rows = q_ptr + tl.where(valid, qrow, b * T * HQ + j * GQ)[:, None] * D
    bt = tl.arange(0, BT)
    # gfx1201_fp8_kv.DECODE_PAIRS reads the fp8 cache as 16-bit pairs (in uint16 units)
    k16 = k_ptr.to(tl.pointer_type(tl.uint16))
    v16 = v_ptr.to(tl.pointer_type(tl.uint16))
    dq2 = tl.arange(0, QC // 2)
    d2 = tl.arange(0, D // 2)
    m_i = tl.full((R,), float("-inf"), tl.float32)
    l_i = tl.zeros((R,), tl.float32)
    acc = tl.zeros((R, D), tl.float32)
    for t in range(t0, t1, BT):
        tt = t + bt
        tok = tl.load(req_to_token + rp * r2t_stride + tl.minimum(tt, t1 - 1))
        # int32 offsets (the caller checks the pool size); 16-byte hints, unmasked loads
        base = tl.multiple_of(tok * (HKV * D) + j * D, 16)
        # q is re-read (from cache) in QC-wide chunks every tile: kept in registers as a
        # WMMA operand it takes 128 VGPRs next to the accumulator's 128 and spills, and
        # the spill reloads queue behind the K/V gathers. volatile keeps it in the loop.
        sc = tl.zeros((R, BT), tl.float32)
        for c in tl.static_range(0, D, QC):
            qc = tl.load(q_rows + c + dq[None, :], volatile=True)
            # bf16 operands for both dots; a no-op for a bf16 cache, the dequant of an
            # fp8 one (gfx1201_fp8_kv; the caller folds k_scale into scale)
            if DECODE == 2:
                kc = gfx1201_fp8_kv.e4m3fn_pairs_to_bf16(
                    tl.load(
                        tl.max_contiguous(
                            tl.multiple_of(
                                k16 + (base // 2)[:, None] + c // 2 + dq2[None, :], [1, 8]
                            ),
                            [1, 8],
                        )
                    )
                )
            elif DECODE == 1:
                kc = gfx1201_fp8_kv.e4m3fn_to_bf16(
                    tl.load(
                        tl.max_contiguous(
                            tl.multiple_of(k_ptr + base[:, None] + c + dq[None, :], [1, 16]),
                            [1, 16],
                        )
                    )
                )
            else:
                kc = tl.load(
                    tl.max_contiguous(
                        tl.multiple_of(k_ptr + base[:, None] + c + dq[None, :], [1, 16]),
                        [1, 8],
                    )
                ).to(tl.bfloat16)
            sc = tl.dot(qc, tl.trans(kc), sc)
        sc = sc * scale
        off = base[:, None] + d[None, :]
        sc = tl.where(
            (tt[None, :] < seq_r[:, None]) & (tt[None, :] < t1), sc, float("-inf")
        )
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(sc - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if DECODE == 2:
            v = gfx1201_fp8_kv.e4m3fn_pairs_to_bf16(
                tl.load(
                    tl.max_contiguous(
                        tl.multiple_of(v16 + (base // 2)[:, None] + d2[None, :], [1, 8]),
                        [1, 8],
                    )
                )
            )
        elif DECODE == 1:
            v = gfx1201_fp8_kv.e4m3fn_to_bf16(
                tl.load(tl.max_contiguous(tl.multiple_of(v_ptr + off, [1, 16]), [1, 16]))
            )
        else:
            v = tl.load(
                tl.max_contiguous(tl.multiple_of(v_ptr + off, [1, 16]), [1, 8])
            ).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    # partial row of (request token row, q head, split)
    prow = qrow * NSPLIT + s
    tl.store(part_ptr + prow[:, None] * D + d[None, :], acc, mask=valid[:, None])
    tl.store(ml_ptr + prow * 2, m_i, mask=valid)
    tl.store(ml_ptr + prow * 2 + 1, l_i, mask=valid)


@triton.jit
def _verify_combine_kernel(
    part_ptr, ml_ptr, o_ptr, v_scale, D: tl.constexpr, NSPLIT: tl.constexpr
):
    """Merge one (token row, q head)'s splits into ``o`` (bf16); ``v_scale`` dequantises
    the fp8 V contribution (1.0 for a bf16 cache)."""
    row = tl.program_id(0)
    d = tl.arange(0, D)
    mx = tl.full((8,), float("-inf"), tl.float32)
    for c in tl.static_range(0, NSPLIT, 8):
        s = c + tl.arange(0, 8)
        mx = tl.maximum(
            mx,
            tl.load(
                ml_ptr + (row * NSPLIT + s) * 2, mask=s < NSPLIT, other=float("-inf")
            ),
        )
    mx = tl.max(mx, axis=0)
    num = tl.zeros((D,), tl.float32)
    den = tl.zeros((8,), tl.float32)
    for c in tl.static_range(0, NSPLIT, 8):
        s = c + tl.arange(0, 8)
        smask = s < NSPLIT
        mm = tl.load(ml_ptr + (row * NSPLIT + s) * 2, mask=smask, other=float("-inf"))
        ll = tl.load(ml_ptr + (row * NSPLIT + s) * 2 + 1, mask=smask, other=0.0)
        w = tl.where(mm == float("-inf"), 0.0, tl.exp(mm - mx))
        acc = tl.load(
            part_ptr + (row * NSPLIT + s)[:, None] * D + d[None, :],
            mask=smask[:, None],
            other=0.0,
        )
        num += tl.sum(w[:, None] * acc, axis=0)
        den += w * ll
    tl.store(
        o_ptr + row * D + d,
        (num / tl.sum(den, axis=0) * v_scale).to(o_ptr.dtype.element_ty),
    )


def covered(q, o, k_buffer, v_buffer, num_tokens_per_req, hq, hkv, d) -> bool:
    """Whether ``verify_attention_gfx1151`` takes this layer: gfx115x, bf16 (or an fp8
    cache under gfx1201_fp8_kv), at most 8 draft tokens per request and 8 q heads per kv
    head, dense ``[B * T, H, D]`` q and o (the kernel addresses them without strides),
    int32-addressable contiguous caches."""
    return (
        enabled()
        and 0 < num_tokens_per_req <= 8
        and hq % hkv == 0
        and hq // hkv <= 8
        and d in (64, 128, 256)
        and q.dtype == torch.bfloat16
        and q.is_contiguous()
        and o.is_contiguous()
        and (
            k_buffer.dtype == v_buffer.dtype == torch.bfloat16
            or gfx1201_fp8_kv.applies(k_buffer, v_buffer)
        )
        and k_buffer.is_contiguous()
        and v_buffer.is_contiguous()
        and k_buffer.shape == v_buffer.shape
        and k_buffer.numel() < 2**31
    )


def verify_attention_gfx1151(
    q,
    o,
    k_buffer,
    v_buffer,
    req_to_token,
    req_pool_indices,
    seq_lens,
    num_tokens_per_req,
    scale,
    nsplit=None,
    bt=None,
    v_scale=1.0,
    num_warps=None,
    hg=6,
    qc=None,
):
    """``o[b * T + t, h]`` = attention of ``q[b * T + t, h]`` over ``seq_lens[b] + t + 1``
    tokens of request ``b`` (``req_to_token[req_pool_indices[b]]``); ``q`` and ``o`` are
    ``[B * T, HQ, D]``, the caches ``[slots, HKV, D]``. For an fp8 cache ``scale``
    includes ``k_scale`` and ``v_scale`` dequantises V. Graph-capturable."""
    rows, hq, d = q.shape
    hkv = k_buffer.shape[1]
    T = num_tokens_per_req
    B = rows // T
    if nsplit is None:
        nsplit = NSPLIT_GFX1201 if gfx1201_enabled() else NSPLIT
    decode = gfx1201_fp8_kv.decode(q, k_buffer, v_buffer, pairs=True)
    if B <= 2 and decode == gfx1201_fp8_kv.DECODE_PAIRS and gfx1201_b1_tile_enabled():
        tile = GFX1201_FP8_B1_TILE
    elif B == 1 and gfx1201_b1_tile_enabled():
        tile = GFX1201_B1_TILE
    else:
        tile = (32, 4, 32)
    bt = tile[0] if bt is None else bt
    num_warps = tile[1] if num_warps is None else num_warps
    qc = tile[2] if qc is None else qc
    if (B, T, hq, hkv) not in _routed:
        _routed.add((B, T, hq, hkv))
        logger.info(
            "Verify attention (gfx115x kernel) route: %d requests x %d tokens, %d q / %d kv heads, %d splits, "
            "tile BT %d / %d warps / QC %d, kv decode %d",
            B, T, hq, hkv, nsplit, bt, num_warps, qc, decode,
        )
    part = torch.empty(rows * hq * nsplit, d, device=q.device, dtype=torch.float32)
    ml = torch.empty(rows * hq * nsplit, 2, device=q.device, dtype=torch.float32)
    ng = -(-(hq // hkv) // hg)
    _verify_split_kernel[(B * hkv * ng * nsplit,)](
        q,
        k_buffer,
        v_buffer,
        part,
        ml,
        req_to_token,
        req_to_token.stride(0),
        req_pool_indices,
        seq_lens,
        scale,
        T=T,
        HQ=hq,
        HKV=hkv,
        D=d,
        NSPLIT=nsplit,
        BT=bt,
        HG=hg,
        QC=qc,
        DECODE=decode,
        num_warps=num_warps,
        # the default software pipelining doubles the LDS and adds spills
        num_stages=1,
    )
    _verify_combine_kernel[(rows * hq,)](
        part, ml, o, float(v_scale), D=d, NSPLIT=nsplit, num_warps=4
    )
    return o
