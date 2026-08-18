#!/usr/bin/env python
"""Drive the real SCMlp / SCAttention band dispatch and check it against the parent.

The unit tests reimplement the band loop against raw ``sc_matmul``; this runs
the actual ``_sc_linear_dynamic_mp`` methods the model calls, so a mistake in
the dispatch itself (row indexing, accumulation, config selection) shows up
here and not only in production.

Two gates:

1. **Parent equivalence.** With ``L[b][k] == stoc_len_levels[k]`` for every
   band, the banded path must reproduce the unbanded per-row path.  Not bit
   identity -- splitting the contraction axis changes only the fp32 summation
   order, since each chunk keeps its own scale and RNG table.

2. **Non-degeneracy.** A tilted ladder must actually move the output.  A band
   dispatch that silently collapses back to the parent would pass gate 1 while
   measuring nothing.
"""
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scmp_kernels.mp import AdaptiveMPConfig
from scmp_kernels.sc.config_helpers import make_sobol_simple_config
from qdit.sc_integration.sc_mlp import SCMlp
from qdit.sc_integration.sc_attention import SCAttention

LEVELS = [128, 96, 64, 48, 32, 16]
CHUNK_D = 128
WIDTH = 1152                      # DiT-XL/2 hidden -> 9 chunks
CHUNK_BANDS = [0, 0, 0, 1, 1, 2, 2, 3, 3]   # widths 384/256/256/256
TOTAL_BLOCKS = 28


class StubController:
    """Only the surface ``_sc_linear_dynamic_mp`` actually reads."""

    def __init__(self, adaptive_mp_config):
        self.adaptive_mp_config = adaptive_mp_config
        self.mp_config = None
        self.current_timestep = 10
        self.total_timesteps = 250
        self.total_blocks = TOTAL_BLOCKS
        self.sc_prec = 8
        self.stoc_len = 128
        self.noise_model = False
        self.halve = True
        self.fixed_level_sc_prec = True

    def resolve_sc_prec(self, stoc_len):
        return self.sc_prec


class StubModule:
    """Duck-types the parts of SCMlp / SCAttention the dispatch touches."""

    def __init__(self, controller, block_idx=3):
        self.sc_controller = controller
        self.block_idx = block_idx
        self.sc_mode = "bipolar"
        self._sc_configs = {}

    _get_sc_config = SCMlp._get_sc_config
    _get_matmul_fn = SCMlp._get_matmul_fn
    _rng_levels = SCMlp._rng_levels


def make_config(band_ladders):
    """An AdaptiveMPConfig whose thresholds spread rows over every rung."""
    n = len(LEVELS)
    thresholds = [round(1.0 - (i + 1) / n, 4) for i in range(n - 1)]
    payload = {
        "stoc_len_levels": LEVELS,
        "timestep_buckets": 1,
        "layer_buckets": 1,
        "operator_defaults": {
            op: {"thresholds": thresholds}
            for op in ("mlp_fc1", "proj")
        },
        "k_bands": {
            "n_bands": 4,
            "chunk_d": CHUNK_D,
            "residual_width": {"mlp_fc1": WIDTH, "proj": WIDTH},
            "chunk_bands": {f"{op}:{b}": CHUNK_BANDS
                            for op in ("mlp_fc1", "proj")
                            for b in range(TOTAL_BLOCKS)},
            "ladders": {f"{op}:t0:l0": band_ladders
                        for op in ("mlp_fc1", "proj")},
        },
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    try:
        return AdaptiveMPConfig(stoc_len_levels=list(LEVELS),
                                threshold_table_path=path)
    finally:
        os.unlink(path)


def run(method, cfg, x, w, operator, banded, **kwargs):
    saved = cfg.k_band_count
    if not banded:
        cfg.k_band_count = 0
    try:
        mod = StubModule(StubController(cfg))
        return method(mod, x, w, None, operator, chunk_d=CHUNK_D, **kwargs)
    finally:
        cfg.k_band_count = saved


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def check(name, banded, parent, *, expect_close):
    r = rel(banded, parent)
    if expect_close:
        ok = r < 1e-5
        verdict = "OK" if ok else "FAIL"
        print(f"[{verdict}] {name}: rel err vs parent {r:.3e} (want < 1e-5)")
    else:
        ok = r > 1e-4
        verdict = "OK" if ok else "FAIL"
        print(f"[{verdict}] {name}: rel err vs parent {r:.3e} (want > 1e-4)")
    return ok


def main():
    if not torch.cuda.is_available():
        print("no CUDA; SC kernels cannot run")
        return 1

    torch.manual_seed(0)
    M, N = 128, 1152
    x = torch.randn(M, WIDTH, device="cuda") * 0.05
    w_mlp = torch.randn(4608, WIDTH, device="cuda") * 0.02
    w_proj = torch.randn(N, WIDTH, device="cuda") * 0.02

    parent_ladders = [list(LEVELS) for _ in range(4)]
    # 2a + b == 3 * parent per rung for widths 384/256/256/256 is
    # (1/3)a0 + (2/9)(a1+a2+a3) == parent; keep it simple and provably legal by
    # trading the widest band down against the narrow ones.
    tilted = [
        [128, 72, 48, 36, 24, 12],
        [128, 108, 72, 54, 36, 18],
        [128, 108, 72, 54, 36, 18],
        [128, 108, 72, 54, 36, 18],
    ]

    cfg_parent = make_config(parent_ladders)
    cfg_tilt = make_config(tilted)

    ok = True
    for label, method, weight, operator, kwargs in [
        ("SCMlp.mlp_fc1", SCMlp._sc_linear_dynamic_mp, w_mlp, "mlp_fc1", {}),
        ("SCAttention.proj", SCAttention._sc_linear_dynamic_mp, w_proj, "proj",
         {"grouped": True}),
    ]:
        parent = run(method, cfg_parent, x, weight, operator, False, **kwargs)
        banded = run(method, cfg_parent, x, weight, operator, True, **kwargs)
        ok &= check(f"{label} parent-ladder", banded, parent, expect_close=True)

        tilt = run(method, cfg_tilt, x, weight, operator, True, **kwargs)
        ok &= check(f"{label} tilted-ladder", tilt, parent, expect_close=False)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
