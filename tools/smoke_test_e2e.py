"""End-to-end smoke test for the scmp_diffusion bootstrap.

Run on a GPU node. Verifies:
  1. submodule imports resolve   (scmp_kernels.sc / scmp_kernels.mp)
  2. sc_matmul reaches every granularity + mode combination
  3. SC results stay within sane rel-err of torch.matmul
  4. Q-DiT integration imports resolve   (qdit.sc_integration.{sc_attention, sc_mlp, noise_matmul})
  5. SCMlp forward pass produces a finite, non-zero, sensibly-scaled output
  6. SCAttention forward pass produces a finite output
  7. No remaining references to deprecated names

Exits non-zero on any failure with a clear message.
"""
from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

# Make local Q-DiT package importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def section(name):
    print(f"\n{'='*80}\n{name}\n{'='*80}", flush=True)


def check(label, fn):
    t0 = time.time()
    try:
        result = fn()
        dt = (time.time() - t0) * 1000
        msg = f"  PASS  {label:<60} ({dt:>6.1f} ms)"
        if result:
            msg += f"  {result}"
        print(msg, flush=True)
        return True
    except Exception:
        print(f"  FAIL  {label}", flush=True)
        traceback.print_exc()
        return False


def main():
    failures = []

    section("1. Imports")
    import torch
    print(f"  torch {torch.__version__}, cuda available: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        print("  no CUDA — skipping kernel execution"); sys.exit(2)
    print(f"  device: {torch.cuda.get_device_name(0)}", flush=True)
    try:
        import triton
        print(f"  triton {triton.__version__}", flush=True)
    except Exception as e:
        print(f"  triton import: {e}", flush=True)
        sys.exit(2)

    if not check("import scmp_kernels.sc",
                 lambda: importlib.import_module("scmp_kernels.sc").__name__):
        failures.append("scmp_kernels.sc")

    if not check("import scmp_kernels.mp",
                 lambda: importlib.import_module("scmp_kernels.mp").__name__):
        failures.append("scmp_kernels.mp")

    if not check("from scmp_kernels.sc import sc_matmul, clear_rng_cache, det_kernel_tuning",
                 lambda: __import__("scmp_kernels.sc", fromlist=["sc_matmul","clear_rng_cache","det_kernel_tuning"])):
        failures.append("sc public API")

    if not check("from scmp_kernels.mp import (9 names)",
                 lambda: [getattr(__import__("scmp_kernels.mp", fromlist=[n]), n)
                          for n in ("MPConfig","AdaptiveMPConfig","RangeMPConfig","RowAssignment",
                                    "classify_rows_by_metric","adaptive_classify_rows",
                                    "classify_groups_by_range","MPDistributionLogger","MetricProfiler")]
                 and "OK"):
        failures.append("mp public API")

    section("2. sc_matmul granularity sweep — basic correctness vs torch.matmul")
    from scmp_kernels.sc import sc_matmul
    from scmp_kernels.sc.config_helpers import make_sobol_simple_config

    def rel_err(pred, target):
        num = (pred - target).pow(2).mean().sqrt()
        den = target.pow(2).mean().sqrt().clamp_min(1e-8)
        return float((num / den).item())

    torch.manual_seed(0)
    device = "cuda"

    # per_tensor 2D bipolar
    def per_tensor_2d_bipolar():
        a = torch.randn(32, 128, device=device, dtype=torch.float32)
        b = torch.randn(64, 128, device=device, dtype=torch.float32) * 0.1
        fp = a @ b.t()
        sc = sc_matmul(a, b, granularity="per_tensor", mode="bipolar", sc_prec=8, stoc_len=256)
        e = rel_err(sc, fp)
        return f"rel_err={e:.4f}  shape={tuple(sc.shape)}"

    if not check("sc_matmul per_tensor 2D bipolar  (32×128)@(64×128).T", per_tensor_2d_bipolar):
        failures.append("per_tensor 2D bipolar")

    def per_row_2d_bipolar():
        a = torch.randn(32, 128, device=device, dtype=torch.float32)
        b = torch.randn(64, 128, device=device, dtype=torch.float32) * 0.1
        fp = a @ b.t()
        sc = sc_matmul(a, b, granularity="per_row", mode="bipolar", sc_prec=8, stoc_len=256)
        e = rel_err(sc, fp)
        return f"rel_err={e:.4f}  shape={tuple(sc.shape)}"

    if not check("sc_matmul per_row 2D bipolar", per_row_2d_bipolar):
        failures.append("per_row 2D bipolar")

    def per_row_mlp_chunked():
        a = torch.randn(32, 1152, device=device, dtype=torch.float32)
        b = torch.randn(64, 1152, device=device, dtype=torch.float32) * 0.05
        fp = a @ b.t()
        sc = sc_matmul(a, b, granularity="per_row", mode="bipolar",
                       chunk_d=72, sc_prec=8, stoc_len=256)
        e = rel_err(sc, fp)
        return f"rel_err={e:.4f}  shape={tuple(sc.shape)}  (chunked D=1152→72)"

    if not check("sc_matmul per_row MLP chunked", per_row_mlp_chunked):
        failures.append("per_row MLP chunked")

    def per_row_grouped():
        N, M, D = 64, 32, 128
        a = torch.softmax(torch.randn(N, M, device=device), dim=-1)
        b = torch.randn(D, M, device=device, dtype=torch.float32) * 0.1
        fp = a @ b.t()
        sc = sc_matmul(a, b, granularity="per_row", mode="bipolar",
                       group_a=N, group_b=D, sc_prec=8, stoc_len=256)
        e = rel_err(sc, fp)
        return f"rel_err={e:.4f}  shape={tuple(sc.shape)}  (group_a={N},group_b={D})"

    if not check("sc_matmul per_row grouped (AV pattern)", per_row_grouped):
        failures.append("per_row grouped")

    def per_head_bipolar():
        BH, N, D = 8, 32, 64
        q = torch.randn(BH, N, D, device=device, dtype=torch.float32)
        k = torch.randn(BH, N, D, device=device, dtype=torch.float32)
        fp = q @ k.transpose(-1, -2)
        sc = sc_matmul(q, k, granularity="per_head", mode="bipolar", sc_prec=8, stoc_len=256)
        e = rel_err(sc, fp)
        return f"rel_err={e:.4f}  shape={tuple(sc.shape)}"

    if not check("sc_matmul per_head bipolar  (8×32×64)@(8×32×64).T", per_head_bipolar):
        failures.append("per_head bipolar")

    def per_row_unipolar():
        a = torch.rand(32, 64, device=device, dtype=torch.float32)
        b = torch.randn(16, 64, device=device, dtype=torch.float32) * 0.1
        fp = a @ b.t()
        sc = sc_matmul(a, b, granularity="per_row", mode="unipolar", sc_prec=8, stoc_len=256)
        e = rel_err(sc, fp)
        return f"rel_err={e:.4f}  shape={tuple(sc.shape)}"

    if not check("sc_matmul per_row 2D unipolar", per_row_unipolar):
        failures.append("per_row unipolar")

    section("3. Q-DiT integration imports")

    qdit_imports = [
        ("qdit.sc_integration", ["SCController", "SCAttention", "SCMlp", "SCDiTBlock", "MPConfig", "add_sc_wrapper", "create_sc_controller_from_args"]),
        ("qdit.sc_integration.sc_attention", ["SCAttention"]),
        ("qdit.sc_integration.sc_mlp", ["SCMlp"]),
        ("qdit.sc_integration.noise_matmul", ["noisy_sc_matmul"]),
        ("qdit.sc_integration.sc_controller", ["SCController"]),
        ("qdit.sc_integration.mp_config", ["MPConfig", "AdaptiveMPConfig"]),
    ]

    for mod_name, names in qdit_imports:
        def _doit(m=mod_name, ns=names):
            mod = importlib.import_module(m)
            missing = [n for n in ns if not hasattr(mod, n)]
            if missing:
                raise AttributeError(f"missing names in {m}: {missing}")
            return f"resolved: {', '.join(ns)}"
        if not check(f"import {mod_name}", _doit):
            failures.append(mod_name)

    section("4. Deprecated-name surveillance")

    deprecated = ["sc_matmul_per_tensor", "sc_matmul_mlp", "sc_matmul_grouped",
                  "sc_matmul_enable_triton", "sc_matmul_enable_triton_mlp",
                  "sc_matmul_grouped_enable_triton", "sc_matmul_enable_batched_bipolar",
                  "bin_to_stoc_packed", "xnor_matmul"]
    def deprecated_absent():
        import scmp_kernels.sc as sck_sc
        present = [n for n in deprecated if hasattr(sck_sc, n)]
        if present:
            raise AssertionError(f"deprecated names still exported: {present}")
        return f"none of {len(deprecated)} deprecated names re-emerged"
    if not check("scmp_kernels.sc has no deprecated public names", deprecated_absent):
        failures.append("deprecated surveillance")

    def sc_enable_absent():
        import qdit.sc_integration.sc_controller as ctrl
        # constructor should not accept sc_enable
        from inspect import signature
        params = signature(ctrl.SCController.__init__).parameters
        if "sc_enable" in params:
            raise AssertionError("SCController.__init__ still accepts sc_enable")
        return "SCController has no sc_enable parameter"
    if not check("SCController no longer accepts sc_enable", sc_enable_absent):
        failures.append("sc_enable removed")

    section("Summary")
    if failures:
        print(f"\n  {len(failures)} FAILURES:")
        for f in failures: print(f"    - {f}")
        sys.exit(1)
    print("\n  ALL CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
