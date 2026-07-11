"""Exhaustive re-verification: scmp_llm/SC vs scmp_kernels.sc.sc_matmul.

For each (granularity, mode) combination, sweeps:
  - 3 different shapes
  - 3 different random seeds
  - 3 different stoc_len values (64, 128, 256)
  - 2 different sc_prec values (6 and 8) where applicable

Every case must report BIT-IDENTICAL (max|Δ|=0). Anything else means a real
divergence remains.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Old impl
SCMP_LLM_SC = Path("/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/SC")
sys.path.insert(0, str(SCMP_LLM_SC))
import sc_triton as old_sc
from config_helpers import make_sobol_simple_config

# New impl
from scmp_kernels.sc import sc_matmul as new_sc_matmul, clear_rng_cache as new_clear

import torch


# Sanity: prove the two impls are genuinely separate modules
print(f"old module:    {old_sc.__file__}", flush=True)
print(f"old enable fn: id={id(old_sc.sc_matmul_enable_triton)}", flush=True)

import scmp_kernels.sc.kernels as new_k
print(f"new module:    {new_k.__file__}", flush=True)
print(f"new dispatch:  id={id(new_sc_matmul)}", flush=True)

assert old_sc.__file__ != new_k.__file__, \
    "scmp_llm and scmp_kernels share a file — comparison would be vacuous"
print()


def reset():
    old_sc.clear_rng_cache()
    new_clear()


fail_count = 0
case_count = 0


def case(label, a, b, *, granularity, mode, sc_prec, stoc_len,
         group_a=1, group_b=1, old_kind="enable_triton"):
    """Run both, compare, increment counters."""
    global fail_count, case_count
    case_count += 1
    D = a.shape[-1]
    cfg = make_sobol_simple_config(D, D, sc_prec)

    reset()
    if old_kind == "enable_triton":
        out_old = old_sc.sc_matmul_enable_triton(
            a, b, a.max().item(), a.min().item(), b.max().item(), b.min().item(),
            mode=mode, sc_prec=sc_prec, config=cfg, stoc_len=stoc_len,
        )
    elif old_kind == "enable_batched_bipolar":
        out_old = old_sc.sc_matmul_enable_batched_bipolar(
            a, b, a.amax(dim=(1, 2)), a.amin(dim=(1, 2)),
            b.amax(dim=(1, 2)), b.amin(dim=(1, 2)),
            sc_prec, cfg, stoc_len=stoc_len,
        )
    elif old_kind == "grouped_enable":
        out_old = old_sc.sc_matmul_grouped_enable_triton(
            a, b, group_a=group_a, group_b=group_b,
            mode=mode, sc_prec=sc_prec, config=cfg, stoc_len=stoc_len,
        )
    else:
        raise ValueError(old_kind)

    reset()
    kw = dict(granularity=granularity, mode=mode,
              sc_prec=sc_prec, stoc_len=stoc_len, config=cfg)
    if granularity == "per_row":
        kw.update(group_a=group_a, group_b=group_b)
    out_new = new_sc_matmul(a, b, **kw)

    max_abs = float((out_old - out_new).abs().max().item())
    ok = (max_abs == 0.0)
    if not ok:
        fail_count += 1
    flag = "✓" if ok else "✗"
    print(f"  {flag}  {label:<70}  max|Δ|={max_abs:.3e}", flush=True)


def main():
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"\n{'─'*100}\nBIPOLAR 2D per_tensor  ─  the path where the clipping fix matters most", flush=True)
    print("─" * 100, flush=True)
    for seed in [0, 1, 42]:
        torch.manual_seed(seed)
        for (N, D, M) in [(8, 32, 8), (32, 128, 64), (128, 1152, 1152)]:
            for sc_prec, stoc_len in [(8, 64), (8, 128), (8, 256), (6, 64)]:
                a = torch.randn(N, D, device="cuda")
                b = torch.randn(M, D, device="cuda") * 0.1
                label = f"seed={seed} N={N:<4} D={D:<5} M={M:<5} sc_prec={sc_prec} stoc_len={stoc_len:<3}"
                case(label, a, b, granularity="per_tensor", mode="bipolar",
                     sc_prec=sc_prec, stoc_len=stoc_len)

    print(f"\n{'─'*100}\nBIPOLAR GEMV per_tensor  (N=1)", flush=True)
    print("─" * 100, flush=True)
    for seed in [0, 1, 42]:
        torch.manual_seed(seed)
        for (D, M) in [(32, 8), (128, 64), (1152, 1152)]:
            for stoc_len in [64, 128, 256]:
                a = torch.randn(1, D, device="cuda")
                b = torch.randn(M, D, device="cuda") * 0.1
                label = f"seed={seed} GEMV  D={D:<5} M={M:<5} sc_prec=8 stoc_len={stoc_len:<3}"
                case(label, a, b, granularity="per_tensor", mode="bipolar",
                     sc_prec=8, stoc_len=stoc_len)

    print(f"\n{'─'*100}\nUNIPOLAR 2D per_tensor", flush=True)
    print("─" * 100, flush=True)
    for seed in [0, 1, 42]:
        torch.manual_seed(seed)
        for (N, D, M) in [(8, 32, 8), (32, 128, 64)]:
            for stoc_len in [128, 256]:
                a = torch.rand(N, D, device="cuda")
                b = torch.randn(M, D, device="cuda") * 0.1
                label = f"seed={seed} unipolar N={N:<4} D={D:<5} M={M:<5} stoc_len={stoc_len:<3}"
                case(label, a, b, granularity="per_tensor", mode="unipolar",
                     sc_prec=8, stoc_len=stoc_len)

    print(f"\n{'─'*100}\nQK per_head bipolar (3D batched)", flush=True)
    print("─" * 100, flush=True)
    for seed in [0, 1, 42]:
        torch.manual_seed(seed)
        for (BH, N, D) in [(4, 16, 32), (8, 64, 64), (16, 256, 72)]:
            for stoc_len in [128, 256]:
                q = torch.randn(BH, N, D, device="cuda")
                k = torch.randn(BH, N, D, device="cuda")
                label = f"seed={seed} QK   BH={BH:<3} N={N:<5} D={D:<5} stoc_len={stoc_len:<3}"
                case(label, q, k, granularity="per_head", mode="bipolar",
                     sc_prec=8, stoc_len=stoc_len, old_kind="enable_batched_bipolar")

    print(f"\n{'─'*100}\nAV grouped per_row bipolar", flush=True)
    print("─" * 100, flush=True)
    for seed in [0, 1, 42]:
        torch.manual_seed(seed)
        for (N, D, M) in [(16, 8, 32), (64, 32, 128), (256, 72, 256)]:
            for stoc_len in [128, 256]:
                attn = torch.softmax(torch.randn(N, M, device="cuda"), dim=-1)
                v = torch.randn(D, M, device="cuda") * 0.1
                label = f"seed={seed} AV   N={N:<4} D={D:<4} M={M:<4} stoc_len={stoc_len:<3}"
                case(label, attn, v, granularity="per_row", mode="bipolar",
                     sc_prec=8, stoc_len=stoc_len,
                     group_a=N, group_b=D, old_kind="grouped_enable")

    print(f"\n{'='*100}", flush=True)
    print(f"Total cases: {case_count}    Bit-identical: {case_count - fail_count}    Diverged: {fail_count}", flush=True)
    print("=" * 100, flush=True)
    if fail_count == 0:
        print("\n  ✓ ALL CASES BIT-IDENTICAL — scmp_llm and scmp_kernels produce equivalent SC outputs.", flush=True)
    else:
        print(f"\n  ✗ {fail_count} cases diverged — investigate.", flush=True)
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
