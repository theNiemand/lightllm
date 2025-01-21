import torch
import triton
import triton.language as tl

from lightllm.common.kernel_config import KernelConfigs
from frozendict import frozendict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from triton import Config


class Fp8MMKernelConfig(KernelConfigs):
    kernel_name: str = "fp8_mm"

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_best_config(
        cls,
        M: int,
        N: int,
        K: int,
        out_dtype: str,
    ) -> dict:
        key_params = {
            "N": N,
            "K": K,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        finded_config = cls.get_the_config(key_params)

        if finded_config:
            # find by M
            config: dict = finded_config[min(finded_config.keys(), key=lambda x: abs(int(x) - M))]
            return config
        else:
            config = {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "GROUP_M": 32,
                "num_warps": 4,
                "num_stages": 3,
            }
        return config

    @classmethod
    def save_config(cls, N: int, K: int, out_dtype: str, config_json: Dict[int, Dict[int, Dict]]):

        key_params = {
            "N": N,
            "K": K,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        return cls.store_config(key_params, config_json)


@triton.jit
def grouped_launch(pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):

    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


def get_cuda_autotune_config():
    return [
        # Good config for fp8 inputs.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def _scaled_gemm(
    A,
    B,
    C,
    Ascale,
    Bscale,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m, pid_n = grouped_launch(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    Ascale_ptrs = Ascale + offs_am
    Bscale_ptrs = Bscale + offs_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    a_s = tl.load(Ascale_ptrs)
    b_s = tl.load(Bscale_ptrs)
    acc = acc * a_s[:, None] * b_s[None, :]
    acc = acc.to(C.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def w8a8_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Ascale: torch.Tensor,
    Bscale: torch.Tensor,
    C: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    **run_config,
) -> torch.Tensor:
    """w8a8fp8 per-token per-channel quantization mm.

    Args:
        A: Matrix A with shape of [M, K].
        B: Matrix B with shape of [K, N].
        Ascale: per-token block-wise Quantization scale for A: [M, ].
        Bscale: Quantization scale for B: [N,].
        C: The output matrix with the shape of [M, N].
        dtype: The data type of C.
    Returns:
        torch.Tensor: C.
    """
    assert A.shape[0] == Ascale.shape[0] and A.shape[-1] == B.shape[0]
    assert A.is_contiguous() and C.is_contiguous()
    M, K = A.shape
    _, N = B.shape
    if not run_config:
        run_config = Fp8MMKernelConfig.try_to_get_best_config(M, N, K, dtype)
    grid = (triton.cdiv(M, run_config["BLOCK_M"]) * triton.cdiv(N, run_config["BLOCK_N"]),)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _scaled_gemm[grid](
        A,
        B,
        C,
        Ascale,
        Bscale,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        # **run_config,
    )

    return C


if __name__ == "__main__":
    import time

    block_size = 128
    output_dtype = torch.bfloat16
    M, N, K = 4096, 256, 7168
    A = torch.randn((M, K), dtype=output_dtype).cuda().to(torch.float8_e4m3fn)  # Activation
    B = torch.randn((K, N), dtype=output_dtype).cuda().to(torch.float8_e4m3fn)  # Weight
    Ascale = torch.randn((M,)).cuda() + 0.2
    Bscale = torch.ones((N,)).cuda()

    C = torch.randn((M, N), dtype=output_dtype).cuda()  # weight
    B = B.T.contiguous().T
    # warmup

    w8a8_fp8_matmul(A, B, Ascale, Bscale, C, output_dtype)

    #### groud truth
    d_A = A.to(output_dtype) * (Ascale.unsqueeze(-1).to(output_dtype))
    d_B = B.to(output_dtype).contiguous()

    gt_C = d_A.mm(d_B)
    # caluate the simlarity
    import torch.nn.functional as F

    cosine_sim = F.cosine_similarity(C.flatten().unsqueeze(0), gt_C.flatten().unsqueeze(0), dim=1)

    print(f"Cosine Similarity between C and gt_C: {cosine_sim.item()}")

    fn2 = lambda: torch.mm(d_A, d_B, out=gt_C)
    ms2 = triton.testing.do_bench(fn2)
    print(f"bf16 time : {ms2} ms")

    fn2 = lambda: w8a8_fp8_matmul(A, B, Ascale, Bscale, C, output_dtype)
    ms2 = triton.testing.do_bench_cudagraph(fn2)
    print(f"fp8 time : {ms2} ms")
