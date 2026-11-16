# SPDX-License-Identifier: Apache-2.0
"""Fused residual add + Gemma RMSNorm + MXFP4 quantization for gfx1201.

One kernel replaces gemma_fused_add_rmsnorm(round_sum=True) (minimax_m3_rmsnorm)
followed by gfx1201_mxfp4.dynamic_mxfp4_quant on the normalized rows. The norm
arithmetic is that kernel's, unchanged (fp32 sum rounded to the activation dtype,
fp32 variance over the row, s * rstd * (1 + w)), launched with the same block
width and warp count, so the normalized BF16 values are the same bits. The
quantization is the vendored AITER _mxfp4_quant_op applied to those BF16 values;
it works per 32-element group, so its output depends only on those values. The
packed values and the transposed E8M0 scales are written in the layout
dynamic_mxfp4_quant produces, which gemm_afp4wfp4 consumes.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.quark.schemes.gfx1201_mxfp4_kernels import _mxfp4_quant_op


def _num_warps(block_n: int) -> int:
    # minimax_m3_rmsnorm._num_warps: the norm's reduction layout follows it.
    if block_n >= 4096:
        return 16
    if block_n >= 1024:
        return 8
    return 4


@triton.jit
def _gemma_add_rmsnorm_mxfp4_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    q_ptr,
    s_ptr,
    res_out_ptr,
    n_cols,
    stride_xrow,
    stride_xcol,
    stride_rrow,
    stride_rcol,
    stride_qrow,
    stride_srow,
    stride_scol,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_xrow + cols * stride_xcol, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + row * stride_rrow + cols * stride_rcol, mask=mask, other=0.0).to(tl.float32)
    s = x + r
    # round_sum=True: normalize the sum as stored.
    s = s.to(res_out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(res_out_ptr + row * n_cols + cols, s.to(res_out_ptr.dtype.element_ty), mask=mask)
    var = tl.sum(s * s, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = s * rstd * (1.0 + w)
    # The separate path stores BF16 and the quantizer reloads it as fp32.
    out = out.to(tl.bfloat16).to(tl.float32)
    q, scales = _mxfp4_quant_op(tl.reshape(out, (1, BLOCK_N)), BLOCK_N, 1, 32)
    q = tl.reshape(q, (BLOCK_N // 2,))
    scales = tl.reshape(scales, (BLOCK_N // 32,))
    qcols = tl.arange(0, BLOCK_N // 2)
    tl.store(q_ptr + row * stride_qrow + qcols, q, mask=qcols < n_cols // 2)
    scols = tl.arange(0, BLOCK_N // 32)
    tl.store(s_ptr + row * stride_srow + scols * stride_scol, scales, mask=scols < n_cols // 32)


def gemma_add_rmsnorm_mxfp4(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    """(x + residual) -> Gemma RMSNorm -> MXFP4.

    Returns ((packed (M, K/2) uint8, scales (M, K/32) uint8 with strides (1, M)), residual_out),
    bitwise equal to dynamic_mxfp4_quant(gemma_fused_add_rmsnorm(..., round_sum=True)[0]) and
    its residual output.
    """
    if x.dim() != 2 or x.dtype != torch.bfloat16 or residual.shape != x.shape:
        raise ValueError("gfx1201 add+norm+MXFP4 takes 2-D BF16 input and a matching residual")
    m, n = x.shape
    if n % 32:
        raise ValueError("hidden size must be a multiple of 32")
    packed = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty_strided((m, n // 32), (1, m), dtype=torch.uint8, device=x.device)
    res_out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_n = triton.next_power_of_2(n)
    _gemma_add_rmsnorm_mxfp4_kernel[(m,)](
        x, residual, weight, packed, scales, res_out, n,
        x.stride(0), x.stride(1), residual.stride(0), residual.stride(1),
        packed.stride(0), scales.stride(0), scales.stride(1), eps,
        BLOCK_N=block_n, num_warps=_num_warps(block_n),
    )
    return (packed, scales), res_out
