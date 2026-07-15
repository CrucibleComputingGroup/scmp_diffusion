"""Compare scmp_kernels.sc.sc_matmul (GPU Triton) against the CPU reference
in scmp_llm/SC/sc_enable.py (sc_matmul_enable).

The CPU reference is the gold-standard semantic check: it uses Python/PyTorch
ops, no Triton, two algorithms:
  - cycle_by_cycle:  exact UnarySim FSUMul simulator (very slow, small shapes)
  - k_shortcut:      vectorized prefix-sum equivalent (faster)

For each shape × mode × stoc_len, runs:
  GPU Triton  (new scmp_kernels)
  CPU k_shortcut  (scmp_llm reference)
  CPU cycle_by_cycle  (scmp_llm reference, on small shapes only)

Compares each pair. Expectations:
  - CPU k_shortcut vs cycle_by_cycle:  should be bit-identical (same math).
  - GPU vs CPU:  should be bit-identical OR differ only by float32 rounding
    (≤ 1e-4 absolute, dominated by accumulation order).
"""
from __future__ import annotations
import sys, time
from pathlib import Path

SCMP_LLM_SC = Path("/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/SC")
sys.path.insert(0, str(SCMP_LLM_SC))
import sc_enable as cpu_ref
from config_helpers import make_sobol_simple_config

from scmp_kernels.sc import sc_matmul as gpu_matmul, clear_rng_cache

import torch


def diff(a, b):
    d = (a - b).abs()
    rel = d / b.abs().clamp_min(1e-6)
    return float(d.max().item()), float(d.mean().item()), float(rel.max().item())


def rel_err(pred, target):
    num = (pred - target).pow(2).mean().sqrt()
    den = target.pow(2).mean().sqrt().clamp_min(1e-8)
    return float((num / den).item())


def reset():
    clear_rng_cache()


def section(title):
    print(f"\n{title}\n{'─'*120}", flush=True)
    print(f"  {'CASE':<55} {'max|Δ|':>10}  {'mean|Δ|':>10}  {'max rel':>10}  {'verdict':>20}", flush=True)


def verdict(ma, threshold=1e-4):
    if ma == 0.0: return "BIT-IDENTICAL"
    if ma < threshold: return f"≈identical ({ma:.1e})"
    return f"differ ({ma:.2e})"


def main():
    print(f"device:        {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CPU reference: {cpu_ref.__file__}", flush=True)
    print(f"GPU impl:      scmp_kernels.sc.sc_matmul", flush=True)

    # ----------------------------------------------------------------
    # 1.  CPU k_shortcut  vs  CPU cycle_by_cycle  (both reference)
    # ----------------------------------------------------------------
    section("1. CPU references self-consistent? — k_shortcut vs cycle_by_cycle")
    torch.manual_seed(0)
    for (N, D, M) in [(4, 16, 4), (8, 32, 8)]:   # cycle_by_cycle is slow
        for mode in ["bipolar", "unipolar"]:
            if mode == "unipolar":
                a = torch.rand(N, D, device="cuda")
            else:
                a = torch.randn(N, D, device="cuda")
            b = torch.randn(M, D, device="cuda") * 0.1
            cfg = make_sobol_simple_config(D, D, 8)
            out_cyc = cpu_ref.sc_matmul_enable(
                a, b, a.max().item(), a.min().item(), b.max().item(), b.min().item(),
                mode=mode, sc_prec=8, config=cfg, method="cycle_by_cycle")
            out_ks = cpu_ref.sc_matmul_enable(
                a, b, a.max().item(), a.min().item(), b.max().item(), b.min().item(),
                mode=mode, sc_prec=8, config=cfg, method="k_shortcut")
            ma, me, mr = diff(out_ks, out_cyc)
            print(f"  {f'{mode:<8} N={N:<3} D={D:<4} M={M:<3} sc_prec=8 stoc_len=256':<55} "
                  f"{ma:>10.3e}  {me:>10.3e}  {mr:>10.3e}  {verdict(ma):>20}", flush=True)

    # ----------------------------------------------------------------
    # 2.  GPU Triton  vs  CPU k_shortcut  — bipolar per_tensor
    # ----------------------------------------------------------------
    section("2. GPU Triton vs CPU k_shortcut — bipolar per_tensor")
    for (N, D, M) in [(8, 32, 8), (32, 128, 64), (64, 256, 32)]:
        for stoc_len in [64, 256]:
            torch.manual_seed(0)
            a = torch.randn(N, D, device="cuda")
            b = torch.randn(M, D, device="cuda") * 0.1
            cfg = make_sobol_simple_config(D, D, 8)
            reset()
            out_cpu = cpu_ref.sc_matmul_enable(
                a, b, a.max().item(), a.min().item(), b.max().item(), b.min().item(),
                mode="bipolar", sc_prec=8, config=cfg, method="k_shortcut")
            # CPU ref uses stoc_len = 2**sc_prec = 256 always (no stoc_len arg!)
            # So we only compare at stoc_len=256 for full equivalence
            if stoc_len != 256:
                continue
            reset()
            out_gpu = gpu_matmul(a, b, granularity="per_tensor", mode="bipolar",
                                  sc_prec=8, stoc_len=stoc_len, config=cfg)
            ma, me, mr = diff(out_gpu, out_cpu)
            print(f"  {f'bipolar  N={N:<3} D={D:<4} M={M:<3} stoc_len={stoc_len:<3}':<55} "
                  f"{ma:>10.3e}  {me:>10.3e}  {mr:>10.3e}  {verdict(ma):>20}", flush=True)

    # ----------------------------------------------------------------
    # 3.  GPU vs CPU — unipolar
    # ----------------------------------------------------------------
    section("3. GPU Triton vs CPU k_shortcut — unipolar per_tensor")
    for (N, D, M) in [(8, 32, 8), (16, 128, 32)]:
        torch.manual_seed(0)
        a = torch.rand(N, D, device="cuda")
        b = torch.randn(M, D, device="cuda") * 0.1
        cfg = make_sobol_simple_config(D, D, 8)
        reset()
        out_cpu = cpu_ref.sc_matmul_enable(
            a, b, a.max().item(), a.min().item(), b.max().item(), b.min().item(),
            mode="unipolar", sc_prec=8, config=cfg, method="k_shortcut")
        reset()
        out_gpu = gpu_matmul(a, b, granularity="per_tensor", mode="unipolar",
                              sc_prec=8, stoc_len=256, config=cfg)
        ma, me, mr = diff(out_gpu, out_cpu)
        print(f"  {f'unipolar N={N:<3} D={D:<4} M={M:<3} stoc_len=256':<55} "
              f"{ma:>10.3e}  {me:>10.3e}  {mr:>10.3e}  {verdict(ma):>20}", flush=True)

    # ----------------------------------------------------------------
    # 4.  Numerical accuracy: both vs torch.matmul fp baseline
    # ----------------------------------------------------------------
    section("4. SC vs fp baseline — both impls should match torch.matmul within SC noise band")
    torch.manual_seed(0)
    N, D, M = 32, 128, 64
    a = torch.randn(N, D, device="cuda")
    b = torch.randn(M, D, device="cuda") * 0.1
    fp = a @ b.t()
    cfg = make_sobol_simple_config(D, D, 8)
    reset()
    out_gpu = gpu_matmul(a, b, granularity="per_tensor", mode="bipolar",
                          sc_prec=8, stoc_len=256, config=cfg)
    out_cpu = cpu_ref.sc_matmul_enable(
        a, b, a.max().item(), a.min().item(), b.max().item(), b.min().item(),
        mode="bipolar", sc_prec=8, config=cfg, method="k_shortcut")
    print(f"  bipolar N=32 D=128 M=64 stoc_len=256", flush=True)
    print(f"    rel_err GPU vs fp:  {rel_err(out_gpu, fp):.4f}", flush=True)
    print(f"    rel_err CPU vs fp:  {rel_err(out_cpu, fp):.4f}", flush=True)
    print(f"    rel_err GPU vs CPU: {rel_err(out_gpu, out_cpu):.4e}", flush=True)


if __name__ == "__main__":
    main()
