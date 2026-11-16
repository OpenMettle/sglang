# SPDX-License-Identifier: MIT
# Derived from AMD's AITER Triton FP4 GEMM (_gemm_afp4wfp4_kernel, MIT), as vendored
# unchanged in gfx1201_mxfp4_kernels.py. Generated from that source by renaming the
# kernel, adding y_ptr/counter_ptr/stride_ym/stride_yn, and replacing only the final
# partial store with a fused split-K reduction (see the epilogue). The GEMM main loop
# is textually identical to the vendored kernel.
"""Split-K MXFP4 GEMM for gfx1201 with the partial reduction fused into the epilogue."""

import triton
import triton.language as tl

from sglang.srt.layers.quantization.quark.schemes.gfx1201_mxfp4_kernels import pid_grid, remap_xcd  # noqa: F401


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0),
    }
)
@triton.jit
def _gemm_afp4wfp4_splitk_fused_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    y_ptr,
    counter_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_ym,
    stride_yn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsk > 0)
    tl.assume(stride_bsn > 0)

    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    # remap so that XCDs get continous chunks of pids (of CHUNK_SIZE).
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)

    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)
    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:

        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

        # Create pointers for first block of A and B input matrices
        # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
        offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
        offs_k_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )
        # Create pointers for the first block of A and B scales
        offs_ks = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)) + tl.arange(
            0, BLOCK_SIZE_K // SCALE_GROUP_SIZE
        )
        a_scale_ptrs = (
            a_scales_ptr + offs_am[:, None] * stride_asm + offs_ks[None, :] * stride_ask
        )
        # B scales are N x K even though B operand is K x N.
        b_scale_ptrs = (
            b_scales_ptr + offs_bn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        offs_scale_k = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            # Load scales, masking OOB when K is not aligned to BLOCK_SIZE_K.
            # OOB scales could contain 0xFF (NaN in e8m0) which propagates
            # through dot_scaled even when the corresponding data is zero.
            if EVEN_K:
                a_scales = tl.load(a_scale_ptrs)
                b_scales = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
            else:
                scale_mask = offs_scale_k[None, :] < (
                    2 * K // SCALE_GROUP_SIZE - k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
                a_scales = tl.load(a_scale_ptrs, mask=scale_mask, other=127)
                b_scales = tl.load(
                    b_scale_ptrs,
                    mask=scale_mask,
                    other=127,
                    cache_modifier=cache_modifier,
                )

            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K // 2), other=0
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < K - k * (BLOCK_SIZE_K // 2),
                    other=0,
                    cache_modifier=cache_modifier,
                )

            accumulator = tl.dot_scaled(
                a, a_scales, "e2m1", b, b_scales, "e2m1", accumulator
            )

            # Advance the ptrs to the next K block.
            a_ptrs += (BLOCK_SIZE_K // 2) * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
            a_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
            b_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        # Fused split-K reduce (bitwise equal to _splitk_reduce_kernel's fixed order).
        # 1. Every wave stores its FP32 partial; the workgroup barrier is preceded by
        #    s_wait_storecnt 0 in every wave (checked in the compiled code by
        #    gfx1201_mxfp4.fused_splitk_asm_ok), so all of this partition's stores are
        #    complete, and on gfx120x a completed store is device-coherent.
        # 2. One agent-scope acq_rel atomic counts the tile's arrivals.
        # 3. The last partition re-reads every partial with device-scope loads
        #    (.cg -> scope:SCOPE_DEV) and sums them as ((p0 + p1) + p2) + ...,
        #    then resets the tile counter for the next launch on the stream.
        tl.store(c_ptrs, c, mask=c_mask)
        tl.debug_barrier()
        arrived = tl.atomic_add(counter_ptr + pid, 1, sem="acq_rel", scope="gpu")
        if arrived == NUM_KSPLIT - 1:
            p_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            acc = tl.load(p_ptrs, mask=c_mask, other=0.0, cache_modifier=".cg")
            for i in tl.static_range(1, NUM_KSPLIT):
                acc = acc + tl.load(p_ptrs + i * stride_ck, mask=c_mask, other=0.0, cache_modifier=".cg")
            y_ptrs = y_ptr + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
            tl.store(y_ptrs, acc.to(y_ptr.type.element_ty), mask=c_mask)
            tl.atomic_xchg(counter_ptr + pid, 0, sem="relaxed", scope="gpu")
