"""Numerical comparison: scmp_llm/SC/sc_triton.py (old) vs scmp_kernels.sc.sc_matmul (new).

For each shape pattern and mode, runs both implementations on identical inputs
with identical Sobol config + identical stoc_len, then reports:
  - max abs diff between the two SC outputs
  - max rel diff between the two SC outputs
  - rel_err of each vs torch.matmul (the fp baseline)

Expected outcome: outputs differ slightly due to the clipping margin removal
(bipolar ±125→±127, unipolar [2,253]→[0,255]). Both should be within sane
SC noise distance from the fp baseline.
"""
from __future__ import annotations
import sys, os
from pathlib import Path

# Old impl — bare imports relative to SC/ folder
SCMP_LLM_SC = Path("/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/SC")
sys.path.insert(0, str(SCMP_LLM_SC))
import sc_triton as old_sc
from config_helpers import make_sobol_simple_config

# New impl — installed as scmp_kernels package
from scmp_kernels.sc import sc_matmul as new_sc_matmul, clear_rng_cache
import scmp_kernels.sc.kernels as new_sc_kernels

import torch


DEVICE = "cuda"


def diff(a, b):
    """Return (max_abs, max_rel)."""
    d = (a - b).abs()
    rel = d / b.abs().clamp_min(1e-6)
    return float(d.max().item()), float(rel.max().item())


def rel_err(pred, target):
    num = (pred - target).pow(2).mean().sqrt()
    den = target.pow(2).mean().sqrt().clamp_min(1e-8)
    return float((num / den).item())


def line(label, vals):
    print(f"  {label:<60}  " + "  ".join(f"{v:>10}" for v in vals), flush=True)


def header(title):
    print(f"\n{title}\n{'='*120}", flush=True)


def reset():
    old_sc.clear_rng_cache()
    clear_rng_cache()


def run_case(name, a, b, *, mode, granularity_new, old_call, sc_prec=8, stoc_len=256):
    """Run both impls on (a, b), print diff vs each other + each vs fp."""
    D = a.shape[-1]
    cfg = make_sobol_simple_config(D, D, sc_prec)
    fp = a @ b.transpose(-1, -2) if b.dim() == a.dim() else a @ b.t()

    reset()
    out_old = old_call(a, b, cfg, sc_prec, stoc_len, mode)

    reset()
    out_new = new_sc_matmul(a, b, granularity=granularity_new, mode=mode,
                            sc_prec=sc_prec, stoc_len=stoc_len, config=cfg)

    assert out_old.shape == out_new.shape == fp.shape, \
        f"{name}: shape mismatch  old={out_old.shape} new={out_new.shape} fp={fp.shape}"

    max_abs, max_rel = diff(out_new, out_old)
    re_old = rel_err(out_old, fp)
    re_new = rel_err(out_new, fp)
    bitwise = "BIT-IDENTICAL" if max_abs == 0 else ("≈identical" if max_abs < 1e-5 else "differ")
    line(name, [f"{max_abs:.3e}", f"{max_rel:.3e}",
                f"{re_old:.4f}", f"{re_new:.4f}", bitwise])


# --------------------------------------------------------------------------
# Adapters — each wraps an old-impl call to a uniform signature
# --------------------------------------------------------------------------
def _old_enable_triton(a, b, cfg, sc_prec, stoc_len, mode):
    return old_sc.sc_matmul_enable_triton(
        a, b, a.max().item(), a.min().item(), b.max().item(), b.min().item(),
        mode=mode, sc_prec=sc_prec, config=cfg, stoc_len=stoc_len,
    )


def _old_enable_batched_bipolar(q, k, cfg, sc_prec, stoc_len, mode):
    q_maxs = q.amax(dim=(1, 2))
    q_mins = q.amin(dim=(1, 2))
    k_maxs = k.amax(dim=(1, 2))
    k_mins = k.amin(dim=(1, 2))
    return old_sc.sc_matmul_enable_batched_bipolar(
        q, k, q_maxs, q_mins, k_maxs, k_mins,
        sc_prec, cfg, stoc_len=stoc_len,
    )


def _old_grouped_enable(a, b, cfg, sc_prec, stoc_len, mode):
    return old_sc.sc_matmul_grouped_enable_triton(
        a, b, group_a=a.shape[0], group_b=b.shape[0],
        mode=mode, sc_prec=sc_prec, config=cfg, stoc_len=stoc_len,
    )


def main():
    print(f"scmp_llm SC source: {SCMP_LLM_SC}", flush=True)
    print(f"new scmp_kernels:   {Path(new_sc_kernels.__file__).parent}", flush=True)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    line("CASE", ["max|Δ|", "max rel-Δ", "rel_err old", "rel_err new", "vs old"])
    print("-" * 120, flush=True)

    torch.manual_seed(0)

    # 2D matmul: per_tensor bipolar — sc_matmul_enable_triton vs sc_matmul(granularity="per_tensor")
    header("BIPOLAR — 2D matmul A(N×D) @ B(M×D).T  ─ enable-signal table-lookup")
    line("CASE", ["max|Δ|", "max rel-Δ", "rel_err old", "rel_err new", "vs old"])
    print("-" * 120, flush=True)

    for shape in [(32, 64, 16), (32, 128, 64), (128, 1152, 1152)]:
        N, D, M = shape
        a = torch.randn(N, D, device=DEVICE)
        b = torch.randn(M, D, device=DEVICE) * 0.1
        run_case(f"matmul   N={N:<4} D={D:<5} M={M:<5}  bipolar  per_tensor",
                 a, b, mode="bipolar", granularity_new="per_tensor",
                 old_call=_old_enable_triton)

    # 2D unipolar
    header("UNIPOLAR — 2D matmul A(N×D) @ B(M×D).T")
    line("CASE", ["max|Δ|", "max rel-Δ", "rel_err old", "rel_err new", "vs old"])
    print("-" * 120, flush=True)

    for shape in [(32, 64, 16), (32, 128, 64)]:
        N, D, M = shape
        a = torch.rand(N, D, device=DEVICE)         # unipolar wants non-neg-ish
        b = torch.randn(M, D, device=DEVICE) * 0.1
        run_case(f"matmul   N={N:<4} D={D:<5} M={M:<5}  unipolar per_tensor",
                 a, b, mode="unipolar", granularity_new="per_tensor",
                 old_call=_old_enable_triton)

    # GEMV — rank-1 a (matrix-vector): A(1×D) @ B(M×D).T → (1, M)
    header("GEMV — A(1×D) @ B(M×D).T  ─ same path, just N=1")
    line("CASE", ["max|Δ|", "max rel-Δ", "rel_err old", "rel_err new", "vs old"])
    print("-" * 120, flush=True)

    for D, M in [(64, 16), (128, 64), (1152, 1152)]:
        a = torch.randn(1, D, device=DEVICE)
        b = torch.randn(M, D, device=DEVICE) * 0.1
        run_case(f"GEMV    D={D:<5} M={M:<5}              bipolar  per_tensor",
                 a, b, mode="bipolar", granularity_new="per_tensor",
                 old_call=_old_enable_triton)

    # 3D per-head bipolar — QK pattern in attention
    header("3D BATCHED BIPOLAR — Q(BH×N×D) @ K(BH×N×D).T  ─ per-head SC")
    line("CASE", ["max|Δ|", "max rel-Δ", "rel_err old", "rel_err new", "vs old"])
    print("-" * 120, flush=True)

    for BH, N, D in [(8, 32, 64), (16, 256, 72)]:
        q = torch.randn(BH, N, D, device=DEVICE)
        k = torch.randn(BH, N, D, device=DEVICE)
        run_case(f"QK      BH={BH:<3} N={N:<5} D={D:<5}    bipolar  per_head",
                 q, k, mode="bipolar", granularity_new="per_head",
                 old_call=_old_enable_batched_bipolar)

    # AV-style grouped matmul (per-row groups)
    header("GROUPED — A(N×D) @ B(M×D).T  ─ per-row groups, group_a=N, group_b=M")
    line("CASE", ["max|Δ|", "max rel-Δ", "rel_err old", "rel_err new", "vs old"])
    print("-" * 120, flush=True)

    for N, D, M in [(64, 32, 128), (256, 72, 256)]:
        a = torch.softmax(torch.randn(N, M, device=DEVICE), dim=-1)  # softmax-like
        b = torch.randn(D, M, device=DEVICE) * 0.1
        N2 = a.shape[0]; M2 = b.shape[0]
        # new dispatcher: granularity=per_row + explicit group_a/group_b
        cfg = make_sobol_simple_config(M, M, 8)
        fp = a @ b.t()
        reset()
        out_old = old_sc.sc_matmul_grouped_enable_triton(
            a, b, group_a=N, group_b=D, mode="bipolar", sc_prec=8, config=cfg, stoc_len=256)
        reset()
        out_new = new_sc_matmul(
            a, b, granularity="per_row", group_a=N, group_b=D,
            mode="bipolar", sc_prec=8, stoc_len=256, config=cfg)
        ma, mr = diff(out_new, out_old)
        re_o = rel_err(out_old, fp); re_n = rel_err(out_new, fp)
        bit = "BIT-IDENTICAL" if ma == 0 else ("≈identical" if ma < 1e-5 else "differ")
        line(f"grouped N={N:<4} D={D:<4} M={M:<4}      bipolar  per_row",
             [f"{ma:.3e}", f"{mr:.3e}", f"{re_o:.4f}", f"{re_n:.4f}", bit])

    print("\nLegend:")
    print("  max|Δ|      = max |out_new - out_old|              (0 ⇒ bit-identical)")
    print("  max rel-Δ   = max |out_new - out_old| / |out_old|")
    print("  rel_err     = SC vs fp baseline (lower = closer to fp)")
    print()
    print("Note: any non-zero Δ is explained entirely by the clipping-margin removal")
    print("  (bipolar: ±125 → ±127, unipolar: [2,253] → [0,255]) which uses more")
    print("  quantization levels — so the new version's rel_err should be ≤ the old.")


if __name__ == "__main__":
    main()
