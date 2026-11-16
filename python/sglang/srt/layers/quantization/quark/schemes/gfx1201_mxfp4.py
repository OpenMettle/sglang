# SPDX-License-Identifier: Apache-2.0
"""Dense MXFP4 activation quantizer and FP4 GEMM for gfx1201 without AITER.

gfx1201 (RDNA4, Radeon AI PRO R9700) has no AITER build. The two Triton
kernels in gfx1201_mxfp4_kernels.py are AMD's AITER kernels (MIT), vendored
from ROCm/aiter 32214c295ea2eec1055bdb3467d2eee7289257be by the qualified
gfx1201 serving runtime; that file is byte-identical to the qualified copy.

The launch geometry is the qualified runtime's. The GEMM keeps BLOCK_SIZE_K
and NUM_KSPLIT fixed, so the K accumulation order is the same for every row
count; only the M/N tile and warp count follow the row count.
"""

import torch
import triton

from sglang.srt.layers.quantization.quark.schemes import gfx1201_mxfp4_kernels as _k

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


def gemm_afp4wfp4(x, w, x_scales, w_scales, dtype, y):
    """y[M, N] = dequant(x) @ dequant(w).T for packed FP4 x (M, K/2) and w (N, K/2)."""
    m, packed_k = x.shape
    n, w_packed_k = w.shape
    if w_packed_k != packed_k:
        raise ValueError("packed reduction dimension mismatch")
    if tuple(y.shape) != (m, n):
        raise ValueError("output shape mismatch")
    k = packed_k * 2
    block_m, block_n, warps = (16, 64, 4) if m <= 16 else (128, 128, 8) if m >= 1024 else (64, 64, 4)
    config = dict(_GEMM_CONFIG, BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, num_warps=warps, SPLITK_BLOCK_SIZE=k)
    _k._gemm_afp4wfp4_kernel[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)](
        x, w, y, x_scales, w_scales,
        m, n, packed_k,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),
        0, y.stride(0), y.stride(1),
        x_scales.stride(0), x_scales.stride(1),
        w_scales.stride(0), w_scales.stride(1),
        **config,
    )
    return y
