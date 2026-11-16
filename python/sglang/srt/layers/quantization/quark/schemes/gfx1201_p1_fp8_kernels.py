# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Private P1 preparation: copied GEMM and quantizer bodies are unchanged.
# GEMM-tail.py carries no file header; SGLang attribution is retained above.
# Expansion below is new private code; no hardware qualification is implied.
import triton
import triton.language as tl

@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
    tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if needs_masking:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _per_token_group_quant_8bit(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid to divide zero
    eps,
    # Information for float8
    bit8_min,
    bit8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.

    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _expand_mxfp4_to_block_fp8(
    Packed,
    GroupScales,
    Expanded,
    BlockScales,
    GuardFlags,
    N: tl.constexpr,
    K: tl.constexpr,
):
    """Lossless logical N128/K128 fold; overwrite one guard flag per tile.

    Flag bits: 1 = scale byte zero, 2 = scale byte 255, 4 = spread > 14.
    Invalid tiles do not write expanded bytes or block scales. The caller must
    reject every nonzero flag before using results or entering a timing phase.
    """
    bn = tl.program_id(0)
    bk = tl.program_id(1)
    rows = bn * 128 + tl.arange(0, 128)
    groups = bk * 4 + tl.arange(0, 4)
    scales = tl.load(
        GroupScales + rows[:, None] * (K // 32) + groups[None, :],
        mask=rows[:, None] < N,
        other=127,
    ).to(tl.int32)
    low = tl.min(tl.min(tl.where(rows[:, None] < N, scales, 255), 1), 0)
    high = tl.max(tl.max(tl.where(rows[:, None] < N, scales, 0), 1), 0)
    invalid_zero = tl.max(tl.max((scales == 0).to(tl.int32), 1), 0)
    invalid_reserved = tl.max(tl.max((scales == 255).to(tl.int32), 1), 0)
    flags = invalid_zero | (invalid_reserved << 1) | ((high - low > 14).to(tl.int32) << 2)
    tl.store(GuardFlags + bn * (K // 128) + bk, flags)
    valid = flags == 0
    exponent = tl.minimum(high - 127, low - 127 + 8)
    # Avoid evaluating unsafe shifts for rejected data. These substitute values
    # are never written and do not constitute a rounded expansion or fallback.
    safe_exponent = tl.where(valid, exponent, 0)
    block_scale = ((safe_exponent + 127) << 23).to(tl.float32, bitcast=True)
    tl.store(BlockScales + bn * (K // 128) + bk, block_scale, mask=valid)

    cols = bk * 128 + tl.arange(0, 128)
    packed = tl.load(
        Packed + rows[:, None] * (K // 2) + cols[None, :] // 2,
        mask=rows[:, None] < N,
        other=0,
    ).to(tl.int32)
    code = (packed >> (4 * (cols[None, :] % 2))) & 15
    m = code & 7
    group_scale = tl.load(
        GroupScales + rows[:, None] * (K // 32) + cols[None, :] // 32,
        mask=rows[:, None] < N,
        other=127,
    ).to(tl.int32)
    delta = tl.where(valid, group_scale - 127 - safe_exponent, 0)
    # Exact E4M3FN encoding, including its subnormal lattice 2^-9.
    # Base codes for .5,1,1.5,2,3,4,6 are 48,56,60,64,68,72,76.
    normal_code = tl.where(m == 1, 48, m * 4 + 48) + delta * 8
    twice_magnitude = tl.where(m < 4, m, tl.where(m < 6, (m - 2) * 2, (m - 4) * 4))
    subnormal_code = twice_magnitude << (delta + 8)
    magnitude_code = tl.where(m == 0, 0, tl.where(normal_code >= 8, normal_code, subnormal_code))
    bits = (magnitude_code | ((code & 8) << 4)).to(tl.uint8)
    tl.store(
        Expanded + rows[:, None] * K + cols[None, :],
        bits.to(Expanded.dtype.element_ty, bitcast=True),
        mask=(rows[:, None] < N) & valid,
    )

