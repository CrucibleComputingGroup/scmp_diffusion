"""Isolate the cause of the bipolar per_tensor diff.

Monkey-patches scmp_llm's ``fused_quantize_bipolar`` to use ``q_clip = q_norm``
(no margin). If this is the sole source of the diff, the output must become
bit-identical to scmp_kernels.
"""
from __future__ import annotations
import sys
from pathlib import Path

SCMP_LLM_SC = Path("/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/SC")
sys.path.insert(0, str(SCMP_LLM_SC))
import sc_triton as old_sc
from config_helpers import make_sobol_simple_config

from scmp_kernels.sc import sc_matmul as new_sc_matmul, clear_rng_cache

import torch


# --- Save originals + install patched fused_quantize_bipolar ---------------
_orig_fqb = old_sc.fused_quantize_bipolar


def patched_fused_quantize_bipolar(fp_tensor, abs_max, sc_prec, rng_levels=None):
    """scmp_llm's body, but with q_clip = q_norm (no margin) — matches scmp_kernels."""
    import triton
    rows, cols = fp_tensor.shape
    q_norm = 2 ** (sc_prec - 1) - 1     # 127
    q_clip = q_norm                      # 127  ← was q_norm - 2
    max_rng_val = old_sc._resolve_rng_levels(sc_prec, rng_levels)
    abs_max = max(abs_max, 1e-5)
    scale = abs_max / q_clip
    inv_scale = 1.0 / scale

    boundary = torch.empty(rows, cols, dtype=torch.int16, device=fp_tensor.device)
    sign = torch.empty(rows, cols, dtype=torch.int8, device=fp_tensor.device)
    total = rows * cols
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    old_sc.fused_quant_bipolar_kernel[grid](
        fp_tensor, boundary, sign,
        inv_scale, q_clip, -q_clip, q_clip, max_rng_val,
        rows, cols, BLOCK,
    )
    return boundary, sign, scale


def diff(a, b):
    d = (a - b).abs()
    return float(d.max().item()), float((d / b.abs().clamp_min(1e-6)).max().item())


def rel_err(pred, target):
    num = (pred - target).pow(2).mean().sqrt()
    den = target.pow(2).mean().sqrt().clamp_min(1e-8)
    return float((num / den).item())


def run_pair(name, a, b, *, patch):
    """If patch=True, use the no-margin version of fused_quantize_bipolar."""
    D = a.shape[-1]
    cfg = make_sobol_simple_config(D, D, 8)
    fp = a @ b.t()

    if patch:
        old_sc.fused_quantize_bipolar = patched_fused_quantize_bipolar
    else:
        old_sc.fused_quantize_bipolar = _orig_fqb

    old_sc.clear_rng_cache(); clear_rng_cache()
    out_old = old_sc.sc_matmul_enable_triton(
        a, b, a.max().item(), a.min().item(), b.max().item(), b.min().item(),
        mode="bipolar", sc_prec=8, config=cfg, stoc_len=256,
    )

    old_sc.clear_rng_cache(); clear_rng_cache()
    out_new = new_sc_matmul(a, b, granularity="per_tensor", mode="bipolar",
                             sc_prec=8, stoc_len=256, config=cfg)

    ma, mr = diff(out_new, out_old)
    re_o = rel_err(out_old, fp)
    re_n = rel_err(out_new, fp)
    bit = "BIT-IDENTICAL" if ma == 0.0 else "differ"
    flag = "[PATCHED]" if patch else "[ORIGINAL]"
    print(f"  {flag:<11}  {name:<46}  max|Δ|={ma:.3e}  rel_err: old={re_o:.4f} new={re_n:.4f}  {bit}", flush=True)


def main():
    torch.manual_seed(0)
    print(f"device: {torch.cuda.get_device_name(0)}\n", flush=True)

    shapes = [
        ("matmul N=32  D=64    M=16  ",  (32, 64, 16)),
        ("matmul N=32  D=128   M=64  ",  (32, 128, 64)),
        ("matmul N=128 D=1152  M=1152", (128, 1152, 1152)),
        ("GEMV   N=1   D=64    M=16  ",  (1, 64, 16)),
        ("GEMV   N=1   D=128   M=64  ",  (1, 128, 64)),
        ("GEMV   N=1   D=1152  M=1152", (1, 1152, 1152)),
    ]

    print("ORIGINAL scmp_llm (q_clip = q_norm - 2 = 125 for 8-bit)")
    print("-" * 110, flush=True)
    inputs = []
    for name, (N, D, M) in shapes:
        a = torch.randn(N, D, device="cuda")
        b = torch.randn(M, D, device="cuda") * 0.1
        inputs.append((name, a, b))
        run_pair(name, a, b, patch=False)

    print()
    print("PATCHED scmp_llm (q_clip = q_norm = 127  ⇒ should match scmp_kernels)")
    print("-" * 110, flush=True)
    for name, a, b in inputs:
        run_pair(name, a, b, patch=True)


if __name__ == "__main__":
    main()
