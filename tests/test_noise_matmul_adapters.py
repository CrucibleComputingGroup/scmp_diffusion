"""
Unit test: compare real SC Triton kernels vs noisy surrogate adapter.

For each of the 4 call patterns used in sc_attention.py / sc_mlp.py,
call the real kernel (``scmp_kernels.sc.sc_matmul`` with the appropriate
``granularity``) and the noisy adapter (``noisy_sc_matmul``, same
signature) with IDENTICAL inputs, then compare:
  1. shape, dtype
  2. mean, std (distribution similarity)
  3. RMSE vs exact float matmul (noise level)
  4. wall-clock time

Shapes chosen to match DiT-XL/2 inference at 256x256:
    B=2, H=16, N=256, D=72       (head_dim = 72 in DiT-XL/2)
    mlp dim 1152 → 4608          (typical Q-DiT weight shapes)

Run:
    python -m pytest tests/test_noise_matmul_adapters.py -sv
  or:
    python tests/test_noise_matmul_adapters.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qdit.sc_integration.noise_matmul import noisy_sc_matmul
from scmp_kernels.sc import sc_matmul
from scmp_kernels.sc.config_helpers import make_sobol_simple_config


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def _stats(name, ref, surrogate, exact):
    ref_rmse = (ref - exact).pow(2).mean().sqrt().item()
    sur_rmse = (surrogate - exact).pow(2).mean().sqrt().item()
    return (
        f"{name:42s} "
        f"ref_rmse={ref_rmse:>9.4f}  sur_rmse={sur_rmse:>9.4f}  "
        f"ratio={sur_rmse/max(ref_rmse,1e-9):>5.2f}x  "
        f"ref:mean={ref.mean().item():>+8.3f} std={ref.std().item():>8.3f}  "
        f"sur:mean={surrogate.mean().item():>+8.3f} std={surrogate.std().item():>8.3f}"
    )


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _run_pair(name, ref_kwargs, sur_kwargs, ref_inputs, sur_inputs, exact):
    """Time and compare a (real, surrogate) pair with matching signatures."""
    _sync(); t0 = time.time()
    ref = sc_matmul(*ref_inputs, **ref_kwargs)
    _sync(); t_ref = time.time() - t0

    _sync(); t0 = time.time()
    sur = noisy_sc_matmul(*sur_inputs, **sur_kwargs)
    _sync(); t_sur = time.time() - t0

    assert ref.shape == sur.shape == exact.shape
    assert sur.dtype == torch.float32
    assert torch.isfinite(sur).all()
    print(
        _stats(name, ref, sur, exact),
        f"t_ref={t_ref*1000:>6.1f}ms  t_sur={t_sur*1000:>6.1f}ms  "
        f"speedup={t_ref/max(t_sur,1e-6):>5.1f}x"
    )


def test_sc_matmul_per_tensor_vs_noisy():
    """2D input projection: (BN, D_in) @ (D_out, D_in)^T — granularity=per_tensor."""
    torch.manual_seed(0)
    M, D_in, D_out = 128, 1152, 1152  # DiT-XL/2 hidden_size=1152
    a = torch.randn(M, D_in, device=DEVICE, dtype=DTYPE)
    w = torch.randn(D_out, D_in, device=DEVICE, dtype=DTYPE) * 0.05

    config = make_sobol_simple_config(D_in, D_in, 8)
    exact = a @ w.transpose(-2, -1)

    print()
    for L in [256, 128, 64, 32, 16]:
        sc_prec = {256:8, 128:7, 64:6, 32:5, 16:4}[L]
        kwargs = dict(granularity="per_tensor", mode="bipolar",
                      sc_prec=sc_prec, config=config, stoc_len=L)
        _run_pair(f"sc_matmul per_tensor L={L:>4d}",
                  kwargs, kwargs, (a, w), (a, w), exact)


def test_sc_matmul_per_row_mlp_vs_noisy():
    """MLP fc1: (M, D) @ (D_hidden, D)^T — granularity=per_row, chunk_d=0."""
    torch.manual_seed(1)
    M, D_in, D_out = 128, 1152, 4608
    a = torch.randn(M, D_in, device=DEVICE, dtype=DTYPE)
    w = torch.randn(D_out, D_in, device=DEVICE, dtype=DTYPE) * 0.05

    config = make_sobol_simple_config(D_in, D_in, 8)
    exact = a @ w.transpose(-2, -1)

    print()
    for L in [256, 64, 16]:
        sc_prec = {256:8, 64:6, 16:4}[L]
        kwargs = dict(granularity="per_row", mode="bipolar",
                      sc_prec=sc_prec, config=config,
                      group_a=1, group_b=1, stoc_len=L)
        _run_pair(f"sc_matmul per_row MLP L={L:>4d}",
                  kwargs, kwargs, (a, w), (a, w), exact)


def test_sc_matmul_per_row_grouped_vs_noisy():
    """AV single-head: attn (N, N) @ v^T (D, N) → (N, D)."""
    torch.manual_seed(2)
    N, D = 256, 72
    v = torch.randn(N, D, device=DEVICE, dtype=DTYPE)
    logits = torch.randn(N, N, device=DEVICE, dtype=DTYPE)
    attn = torch.softmax(logits, dim=-1)
    v_t = v.transpose(-2, -1).contiguous()  # (D, N)

    config = make_sobol_simple_config(N, N, 8)
    exact = attn @ v

    print()
    for L in [256, 64, 16]:
        sc_prec = {256:8, 64:6, 16:4}[L]
        kwargs = dict(granularity="per_row",
                      group_a=1, group_b=1,
                      mode="bipolar", sc_prec=sc_prec,
                      config=config, stoc_len=L)
        _run_pair(f"sc_matmul per_row grouped L={L:>4d}",
                  kwargs, kwargs, (attn, v_t), (attn, v_t), exact)


def test_sc_matmul_per_head_vs_noisy():
    """QK batched: (BH, N, D) @ (BH, N, D)^T → (BH, N, N) — granularity=per_head."""
    torch.manual_seed(3)
    B, H, N, D = 2, 16, 256, 72
    q = torch.randn(B*H, N, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B*H, N, D, device=DEVICE, dtype=DTYPE)

    config = make_sobol_simple_config(D, D, 8)
    exact = q @ k.transpose(-2, -1)

    print()
    for L in [256, 64, 16]:
        sc_prec = {256:8, 64:6, 16:4}[L]
        kwargs = dict(granularity="per_head", mode="bipolar",
                      sc_prec=sc_prec, config=config, stoc_len=L)
        _run_pair(f"sc_matmul per_head L={L:>4d}",
                  kwargs, kwargs, (q, k), (q, k), exact)


if __name__ == "__main__":
    print(f"Running on device: {DEVICE}")
    print("=" * 180)
    print("Legend: ref = real SC kernel, sur = noisy surrogate adapter")
    print("        ratio = sur_rmse / ref_rmse (should be ~1.0 if surrogate is calibrated correctly)")
    print("        surrogate should be MUCH faster than real SC")
    print("=" * 180)
    test_sc_matmul_per_tensor_vs_noisy()
    test_sc_matmul_per_row_mlp_vs_noisy()
    test_sc_matmul_per_row_grouped_vs_noisy()
    test_sc_matmul_per_head_vs_noisy()
    print("\nAll tests passed.")
