# SPDX-License-Identifier: Apache-2.0
"""Fused SiLU(gate) * up + MXFP4 quantization for gfx1201 (candidate; bitwise target).

Replaces sgl_hip act_and_mul_kernel<bf16, silu> followed by
gfx1201_mxfp4.dynamic_mxfp4_quant. The HIP kernel computes, per element,
bf16(bf16(g / (1 + expf(-g))) * u) with g, u read as fp32; this kernel uses the
same rounding points, the device library exp (libdevice) and round-to-nearest
division, so the BF16 activations can be the same bits. The quantization is the
vendored AITER _mxfp4_quant_op (per 32-element group).
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice

from sglang.srt.layers.quantization.quark.schemes.gfx1201_mxfp4_kernels import _mxfp4_quant_op


@triton.jit
def _silu_mul_mxfp4_kernel(
    gu_ptr,
    q_ptr,
    s_ptr,
    d,
    stride_gu,
    stride_q,
    stride_srow,
    stride_scol,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < d
    g = tl.load(gu_ptr + row * stride_gu + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * stride_gu + d + cols, mask=mask, other=0.0).to(tl.float32)
    act = tl.math.div_rn(g, 1.0 + libdevice.exp(-g))
    act = act.to(tl.bfloat16).to(tl.float32)
    out = (act * u).to(tl.bfloat16).to(tl.float32)
    q, scales = _mxfp4_quant_op(tl.reshape(out, (1, BLOCK)), BLOCK, 1, 32)
    q = tl.reshape(q, (BLOCK // 2,))
    scales = tl.reshape(scales, (BLOCK // 32,))
    qcols = blk * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    tl.store(q_ptr + row * stride_q + qcols, q, mask=qcols < d // 2)
    scols = blk * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    tl.store(s_ptr + row * stride_srow + scols * stride_scol, scales, mask=scols < d // 32)


def silu_mul_mxfp4(gate_up: torch.Tensor):
    """(packed (M, D/2) uint8, scales (M, D/32) uint8 with strides (1, M)) for SiLU(gate)*up,
    gate_up = [gate | up] of width 2D."""
    if gate_up.dim() != 2 or gate_up.dtype != torch.bfloat16 or gate_up.shape[1] % 64:
        raise ValueError("gfx1201 SiLU*mul+MXFP4 takes 2-D BF16 [gate | up] with D % 32 == 0")
    gate_up = gate_up.contiguous()
    m, d = gate_up.shape[0], gate_up.shape[1] // 2
    packed = torch.empty((m, d // 2), dtype=torch.uint8, device=gate_up.device)
    scales = torch.empty_strided((m, d // 32), (1, m), dtype=torch.uint8, device=gate_up.device)
    block = 1024
    _silu_mul_mxfp4_kernel[(m, triton.cdiv(d, block))](
        gate_up, packed, scales, d, gate_up.stride(0), packed.stride(0), scales.stride(0), scales.stride(1),
        BLOCK=block, num_warps=4,
    )
    return packed, scales
