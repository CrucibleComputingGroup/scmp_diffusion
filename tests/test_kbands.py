"""K-bands: table validation, column partitioning, and numeric equivalence.

The equivalence test is the one that matters: with every band's ladder equal to
the parent ladder, the banded dispatch must reproduce the per-row parent.  That
is what makes the refinement unable to lose, so if it ever stops holding the
whole allocation story is void.
"""
import json
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scmp_kernels.mp import AdaptiveMPConfig  # noqa: E402
from qdit.sc_integration.sc_kbands import band_columns  # noqa: E402


LEVELS = [128, 96, 64, 32]
# DiT-XL/2: hidden 1152 -> 9 chunks of 128 for proj / mlp_fc1.
FC1_WIDTH = 1152
CHUNK_D = 128
# 4 bands over 9 chunks: widths 256/256/256/384 channels.
CHUNK_BANDS = [0, 0, 1, 1, 2, 2, 3, 3, 3]
# Solved so 2a + b == 3 * parent for every rung, which is the iso-compute
# identity for these widths: (2/9)*3a + (1/3)b == parent.
WIDE = [144, 112, 80, 40]
NARROW = [96, 64, 32, 16]


def make_table(*, ladders=None, chunk_bands=None, n_bands=4,
               residual_width=None, chunk_d=CHUNK_D, levels=None):
    levels = LEVELS if levels is None else levels
    payload = {
        "stoc_len_levels": levels,
        "timestep_buckets": 1,
        "layer_buckets": 1,
        "buckets": {
            "mlp_fc1:t0:l0": {"thresholds": [0.75, 0.5, 0.25][: len(levels) - 1]},
        },
        "k_bands": {
            "n_bands": n_bands,
            "chunk_d": chunk_d,
            "residual_width": ({"mlp_fc1": FC1_WIDTH}
                               if residual_width is None else residual_width),
            "chunk_bands": ({"mlp_fc1:0": CHUNK_BANDS}
                            if chunk_bands is None else chunk_bands),
            "ladders": ({"mlp_fc1:t0:l0": [WIDE, WIDE, WIDE, NARROW]}
                        if ladders is None else ladders),
        },
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    return path


def load(**kwargs):
    levels = kwargs.pop("levels", None) or LEVELS
    path = make_table(levels=levels, **kwargs)
    try:
        return AdaptiveMPConfig(stoc_len_levels=list(levels),
                                threshold_table_path=path)
    finally:
        os.unlink(path)


class TestKBandTableValidation(unittest.TestCase):

    def test_loads_and_exposes_bands(self):
        cfg = load()
        self.assertEqual(cfg.k_band_count, 4)
        self.assertEqual(cfg.k_band_chunk_d, CHUNK_D)
        bands, ladders = cfg.get_k_bands("mlp_fc1", 0, 28)
        self.assertEqual(bands, CHUNK_BANDS)
        self.assertEqual(ladders, [WIDE, WIDE, WIDE, NARROW])

    def test_absent_section_leaves_path_disabled(self):
        cfg = AdaptiveMPConfig(stoc_len_levels=list(LEVELS))
        self.assertEqual(cfg.k_band_count, 0)
        self.assertIsNone(cfg.get_k_bands("mlp_fc1", 0, 28))

    def test_uncovered_operator_returns_none(self):
        cfg = load()
        self.assertIsNone(cfg.get_k_bands("mlp_fc2", 0, 28))

    def test_parent_ladder_in_every_band_is_iso_compute(self):
        """L[b][k] == parent[k] must pass the identity exactly."""
        parent_everywhere = [list(LEVELS) for _ in range(4)]
        cfg = load(ladders={"mlp_fc1:t0:l0": parent_everywhere})
        _, ladders = cfg.get_k_bands("mlp_fc1", 0, 28)
        self.assertEqual(ladders, parent_everywhere)

    def test_overspend_is_rejected(self):
        # +16 on one wide band with no compensation elsewhere.
        over = [[160, 112, 80, 40], WIDE, WIDE, NARROW]
        with self.assertRaisesRegex(ValueError, "overspends rung 0"):
            load(ladders={"mlp_fc1:t0:l0": over})

    def test_float_noise_is_not_overspend(self):
        """The identity is exact in reals; only true overspend may raise."""
        cfg = load()  # WIDE/NARROW hit the identity dead on
        self.assertIsNotNone(cfg.get_k_bands("mlp_fc1", 0, 28))

    def test_large_underspend_is_rejected(self):
        half = [[x // 2 for x in WIDE] for _ in range(3)] + [
            [x // 2 for x in NARROW]]
        with self.assertRaisesRegex(ValueError, "underspends rung 0"):
            load(ladders={"mlp_fc1:t0:l0": half})

    def test_band_with_one_chunk_is_rejected(self):
        lonely = [0, 0, 1, 1, 1, 2, 2, 2, 3]  # band 3 owns a single chunk
        with self.assertRaisesRegex(ValueError, "band 3 with 1 chunk"):
            load(chunk_bands={"mlp_fc1:0": lonely})

    def test_wrong_chunk_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "chunk entries"):
            load(chunk_bands={"mlp_fc1:0": CHUNK_BANDS[:-1]})

    def test_ladder_rung_count_must_match_levels(self):
        short = [WIDE[:-1], WIDE[:-1], WIDE[:-1], NARROW[:-1]]
        with self.assertRaisesRegex(ValueError, "rungs, expected 4"):
            load(ladders={"mlp_fc1:t0:l0": short})

    def test_chunk_bands_without_ladders_is_rejected(self):
        """Silently falling back to the parent is the failure to prevent."""
        with self.assertRaisesRegex(ValueError, "no.*ladders entry"):
            load(ladders={})

    def test_missing_residual_width_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "residual_width is required"):
            load(residual_width={})

    def test_band_widths_must_agree_across_blocks(self):
        other = [0, 0, 0, 1, 1, 2, 2, 3, 3]  # same op, different widths
        with self.assertRaisesRegex(ValueError, "band widths"):
            load(chunk_bands={"mlp_fc1:0": CHUNK_BANDS, "mlp_fc1:1": other})

    def test_n_bands_below_two_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "n_bands must be >= 2"):
            load(n_bands=1)


class TestBandColumns(unittest.TestCase):

    class _Module:
        pass

    def test_partitions_the_contraction_axis(self):
        m = self._Module()
        cols, widths = band_columns(m, CHUNK_BANDS, 4, CHUNK_D, FC1_WIDTH,
                                    torch.device("cpu"))
        self.assertEqual(widths, [256, 256, 256, 384])
        self.assertEqual(sum(widths), FC1_WIDTH)
        seen = torch.cat(cols).sort().values
        self.assertTrue(torch.equal(seen, torch.arange(FC1_WIDTH)))

    def test_chunks_stay_ascending_inside_a_band(self):
        """The tail chunk must land last, or a band re-chunks differently."""
        # 1160 channels -> 10 chunks: nine 128-wide plus an 8-wide tail.
        bands = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3]
        m = self._Module()
        cols, widths = band_columns(m, bands, 4, CHUNK_D, 1160,
                                    torch.device("cpu"))
        self.assertEqual(widths[3], 128 + 128 + 128 + 8)
        tail = cols[3]
        self.assertTrue(bool((tail[1:] > tail[:-1]).all()))
        self.assertEqual(int(tail[-1]), 1159)

    def test_cache_hits_on_repeat(self):
        m = self._Module()
        a, _ = band_columns(m, CHUNK_BANDS, 4, CHUNK_D, FC1_WIDTH,
                            torch.device("cpu"))
        b, _ = band_columns(m, CHUNK_BANDS, 4, CHUNK_D, FC1_WIDTH,
                            torch.device("cpu"))
        self.assertIs(a, b)

    def test_non_partition_is_rejected(self):
        m = self._Module()
        bands = [0, 0, 1, 1, 2, 2, 3, 3, 3]
        with self.assertRaisesRegex(ValueError, "does not partition"):
            band_columns(m, bands, 3, CHUNK_D, FC1_WIDTH,
                         torch.device("cpu"))


class TestControllerGuards(unittest.TestCase):
    """The controller rejects allocations it cannot actually execute."""

    def _controller(self, halve=True, sc_prec=8):
        from qdit.sc_integration.sc_controller import SCController
        c = SCController.__new__(SCController)
        c.halve = halve
        c.sc_prec = sc_prec
        c.adaptive_mp_config = None
        return c

    def test_qk_in_the_table_is_rejected(self):
        cfg = load()
        cfg.k_band_chunks[("qk", 0)] = CHUNK_BANDS
        with self.assertRaisesRegex(ValueError, r"no band dispatch.*\['qk'\]"):
            self._controller().init_adaptive_mp(cfg)

    def test_band_above_halve_ceiling_is_rejected(self):
        cfg = load()
        # 144 > 2**(8-1); legal by iso-compute, unrealizable on halve hardware.
        with self.assertRaisesRegex(ValueError, "above the halve-mode maximum"):
            self._controller(halve=True, sc_prec=8).init_adaptive_mp(cfg)

    def test_same_table_passes_without_halve(self):
        cfg = load()
        self._controller(halve=False).init_adaptive_mp(cfg)  # must not raise

    def test_clean_table_passes(self):
        cfg = load(ladders={"mlp_fc1:t0:l0": [list(LEVELS) for _ in range(4)]})
        self._controller(halve=True, sc_prec=8).init_adaptive_mp(cfg)


@unittest.skipUnless(torch.cuda.is_available(), "SC kernels need CUDA")
class TestKBandEquivalence(unittest.TestCase):
    """Banded dispatch with the parent ladder must reproduce the parent.

    Not bit-identity: splitting the contraction axis changes only the order in
    which per-chunk partial products are summed in fp32.  Every chunk keeps its
    own scale and RNG table, so the partials themselves are unchanged.
    """

    def _run(self, band_ladders):
        from scmp_kernels.sc import sc_matmul
        from scmp_kernels.sc.config_helpers import make_sobol_simple_config

        torch.manual_seed(0)
        M, D, N = 64, FC1_WIDTH, 256
        x = torch.randn(M, D, device="cuda")
        w = torch.randn(N, D, device="cuda")
        rungs = torch.randint(0, len(LEVELS), (M,), device="cuda")

        out = torch.zeros(M, N, device="cuda", dtype=torch.float32)
        m = TestBandColumns._Module()
        cols, widths = band_columns(m, CHUNK_BANDS, 4, CHUNK_D, D, x.device)
        for b in range(4):
            xb = x.index_select(1, cols[b]).contiguous()
            wb = w.index_select(1, cols[b]).contiguous()
            for k in range(len(LEVELS)):
                sl = band_ladders[b][k]
                rows = (rungs == k).nonzero(as_tuple=True)[0]
                if rows.numel() == 0 or sl <= 0:
                    continue
                cfg = make_sobol_simple_config(CHUNK_D, CHUNK_D, 8)
                out[rows] += sc_matmul(
                    xb.index_select(0, rows).contiguous(), wb,
                    granularity="per_row", mode="bipolar", sc_prec=8,
                    config=cfg, group_a=1, group_b=1, chunk_d=CHUNK_D,
                    stoc_len=sl)
        return x, w, rungs, out

    def test_parent_ladder_reproduces_unbanded(self):
        from scmp_kernels.sc import sc_matmul
        from scmp_kernels.sc.config_helpers import make_sobol_simple_config

        parent_everywhere = [list(LEVELS) for _ in range(4)]
        x, w, rungs, banded = self._run(parent_everywhere)

        unbanded = torch.zeros_like(banded)
        cfg = make_sobol_simple_config(CHUNK_D, CHUNK_D, 8)
        for k, sl in enumerate(LEVELS):
            rows = (rungs == k).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            unbanded[rows] = sc_matmul(
                x.index_select(0, rows).contiguous(), w,
                granularity="per_row", mode="bipolar", sc_prec=8,
                config=cfg, group_a=1, group_b=1, chunk_d=CHUNK_D,
                stoc_len=sl)

        rel = ((banded - unbanded).norm() / unbanded.norm()).item()
        self.assertLess(rel, 1e-5, f"banded vs parent rel err {rel:.3e}")

    def test_nonuniform_ladder_changes_the_result(self):
        """A real allocation must not be a no-op dressed as one."""
        _, _, _, parent = self._run([list(LEVELS) for _ in range(4)])
        _, _, _, tuned = self._run([WIDE, WIDE, WIDE, NARROW])
        rel = ((tuned - parent).norm() / parent.norm()).item()
        self.assertGreater(rel, 1e-4, "band ladders had no effect")


if __name__ == "__main__":
    unittest.main(verbosity=2)
