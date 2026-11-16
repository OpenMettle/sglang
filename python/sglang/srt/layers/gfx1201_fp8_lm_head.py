# SPDX-License-Identifier: Apache-2.0
"""Block-128 FP8 copy of a dense BF16 LM head for DFLASH rows on gfx1201.

With SGLANG_GFX1201_FP8_LM_HEAD=1 (HIP without AITER), the target model's
pre-capture hook quantizes each dense, unquantized ParallelLMHead once:
FP32 amax / 448 per 128x128 block (2^-20 for an all-zero block), values
rounded to E4M3FN after clamping to +-448. The BF16 weight stays in place for
every other call. Row counts that DFLASH uses then read the FP8 copy, half
the bytes of the BF16 head:

- target verify (LogitsProcessor), 8 or 16 rows, FP32 or BF16 output as the
  processor requests;
- draft candidate projection (models/dflash.py), 7 or 14 rows, output in the
  head weight's dtype.

The draft borrows the target's head module, so both use one FP8 copy. The
activation quantizer and GEMM are upstream's per-token-group 8-bit quantizer
and block FP8 matmul, launched with the op234 runtime's small-row tile.
"""

import logging

import torch
import triton

from sglang.srt.utils import get_bool_env_var, is_hip

logger = logging.getLogger(__name__)

ENABLED = (
    is_hip()
    and not get_bool_env_var("SGLANG_USE_AITER")
    and get_bool_env_var("SGLANG_GFX1201_FP8_LM_HEAD")
)
ROWS = (7, 8, 14, 16)
_BLOCK = 128
_FP8_MAX = 448.0
_GEMM_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "needs_masking": False,
    "num_warps": 4,
    "num_stages": 1,
}
_ATTR = "_gfx1201_fp8_head"
_routed: set = set()


def _quantize_weight(weight: torch.Tensor):
    """Block-128 E4M3FN weight and FP32 (N/128, K/128) scales, one band at a time."""
    n, k = weight.shape
    b8 = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight.device)
    scales = torch.empty((n // _BLOCK, k // _BLOCK), dtype=torch.float32, device=weight.device)
    for row in range(0, n, _BLOCK):
        band = weight[row : row + _BLOCK].to(torch.float32).reshape(_BLOCK, k // _BLOCK, _BLOCK)
        amax = band.abs().amax(dim=(0, 2))
        # FP64 intermediates round the scale and its reciprocal once each.
        scale = (amax.to(torch.float64) / _FP8_MAX).to(torch.float32)
        scale = torch.where(amax == 0, 2.0**-20, scale)
        inverse = scale.to(torch.float64).reciprocal().to(torch.float32)
        scaled = (band * inverse[None, :, None]).clamp_(-_FP8_MAX, _FP8_MAX)
        b8[row : row + _BLOCK].copy_(scaled.reshape(_BLOCK, k).to(torch.float8_e4m3fn))
        scales[row // _BLOCK].copy_(scale)
    return b8, scales


def _eligible_head(module) -> bool:
    from sglang.srt.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
    )

    weight = getattr(module, "weight", None)
    return (
        isinstance(module, ParallelLMHead)
        and isinstance(getattr(module, "quant_method", None), UnquantizedEmbeddingMethod)
        and not hasattr(module, "set_lora")
        and getattr(module, "bias", None) is None
        and isinstance(weight, torch.Tensor)
        and weight.dim() == 2
        and weight.dtype == torch.bfloat16
        and weight.is_cuda
        and weight.is_contiguous()
        and weight.shape[0] % _BLOCK == 0
        and weight.shape[1] % _BLOCK == 0
    )


def prepare_model(model: torch.nn.Module) -> int:
    """Attach the FP8 copy to every eligible head in `model`; idempotent."""
    if not ENABLED or torch.cuda.is_current_stream_capturing():
        return 0
    prepared = 0
    for module in model.modules():
        if getattr(module, _ATTR, None) is not None:
            prepared += 1
            continue
        if not _eligible_head(module):
            continue
        with torch.no_grad():
            b8, scales = _quantize_weight(module.weight.data)
        setattr(module, _ATTR, (module.weight, module.weight.data_ptr(), b8, scales))
        prepared += 1
        logger.info(
            "FP8 LM head (gfx1201) prepared: %s, %.0f MiB FP8 + scales",
            tuple(module.weight.shape),
            (b8.numel() + 4 * scales.numel()) / 2**20,
        )
    return prepared


def project(hidden: torch.Tensor, lm_head, out_dtype: torch.dtype, num_rows=None):
    """FP8 logits [M, N] for eligible DFLASH rows, or None to use the BF16 path.

    num_rows restricts the head to its first rows (the draft's org vocabulary);
    only the whole head is supported, so a smaller value returns None.
    """
    state = getattr(lm_head, _ATTR, None)
    if state is None:
        return None
    weight, pointer, b8, scales = state
    n, k = b8.shape
    if (
        lm_head.weight is not weight
        or weight.data_ptr() != pointer
        or (num_rows is not None and num_rows != n)
        or hidden.dim() != 2
        or hidden.shape[0] not in ROWS
        or hidden.shape[1] != k
        or hidden.dtype != torch.bfloat16
        or not hidden.is_contiguous()
        or out_dtype not in (torch.float32, torch.bfloat16)
    ):
        return None
    from sglang.kernels.ops.gemm.fp8_kernel import _w8a8_block_fp8_matmul
    from sglang.kernels.ops.quantization.fp8_kernel import _per_token_group_quant_8bit

    m = hidden.shape[0]
    if (m, out_dtype) not in _routed:
        _routed.add((m, out_dtype))
        logger.info("FP8 LM head (gfx1201) route: %d rows, %s output", m, out_dtype)
    x_q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=hidden.device)
    x_s = torch.empty((m, k // _BLOCK), dtype=torch.float32, device=hidden.device)
    out = torch.empty((m, n), dtype=out_dtype, device=hidden.device)
    _per_token_group_quant_8bit[(m * (k // _BLOCK),)](
        hidden, x_q, x_s, _BLOCK, _BLOCK, 1e-10, -_FP8_MAX, _FP8_MAX,
        BLOCK=_BLOCK, num_warps=1, num_stages=1,
    )
    _w8a8_block_fp8_matmul[(triton.cdiv(m, _GEMM_CONFIG["BLOCK_SIZE_M"]) * triton.cdiv(n, _BLOCK),)](
        x_q, b8, out, x_s, scales, m, n, k, _BLOCK, _BLOCK,
        x_q.stride(0), x_q.stride(1), b8.stride(1), b8.stride(0),
        out.stride(0), out.stride(1), x_s.stride(0), x_s.stride(1),
        scales.stride(1), scales.stride(0),
        **_GEMM_CONFIG,
    )
    return out
