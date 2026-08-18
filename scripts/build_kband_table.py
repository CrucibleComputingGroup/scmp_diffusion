#!/usr/bin/env python
"""Add a ``k_bands`` section to an existing adaptive MP threshold table.

Row dispatch is untouched: the same calibrated thresholds put each row on the
same rung.  What this adds is a partition of the contraction axis into bands of
whole quantization chunks, and a per-band ladder that runs each rung at its own
stream length under an exact per-rung iso-compute identity::

    sum_b (w_b / R) * L[b][k] == L_parent[k]

Because the parent sits inside that space (``L[b][k] == L_parent[k]`` for every
band), the refinement cannot lose -- a tilt that does not help degenerates back
to the parent rather than costing accuracy.

The tilt here is a straight-line shape across bands, which is a sweepable
starting point, not a solved allocation.  ``--tilt 0`` reproduces the parent
exactly and is the sanity arm every sweep should include.

Usage::

    # sanity arm: must reproduce the per-row parent
    python scripts/build_kband_table.py --in mp_best.json --out mp_kb0.json \\
        --tilt 0 --n-bands 4 --max-stoc-len 128

    # front-loaded: early chunks longer, late chunks shorter
    python scripts/build_kband_table.py --in mp_best.json --out mp_kb.json \\
        --tilt 0.25 --n-bands 4 --max-stoc-len 128
"""
import argparse
import json
import sys

# DiT-XL/2 contraction widths per operator. Override with --width op=N.
DEFAULT_WIDTHS = {
    "mlp_fc1": 1152,
    "mlp_fc2": 4608,
    "proj": 1152,
    "input_proj": 1152,
}


def chunk_widths(residual_width, chunk_d):
    n = (residual_width + chunk_d - 1) // chunk_d
    return [min(chunk_d, residual_width - c * chunk_d) for c in range(n)]


def contiguous_bands(n_chunks, n_bands):
    """Assign chunks to bands in contiguous, as-equal-as-possible runs.

    Contiguous because a tilt is a statement about position along the
    contraction axis; scattered bands would make the shape meaningless.
    Every band needs >= 2 chunks to stay on the chunked kernel path, so the
    caller must cap n_bands at n_chunks // 2.
    """
    if n_bands * 2 > n_chunks:
        raise ValueError(
            f"{n_chunks} chunks cannot be split into {n_bands} bands of >= 2 "
            f"chunks; the widest allocation here is {n_chunks // 2} bands.")
    base, extra = divmod(n_chunks, n_bands)
    bands, chunk = [], 0
    for b in range(n_bands):
        size = base + (1 if b < extra else 0)
        bands.extend([b] * size)
        chunk += size
    return bands


def solve_ladder(parent_levels, widths, tilt, max_stoc_len, snap):
    """Per-band lengths that spend at most the parent's per-rung budget.

    Rounds DOWN off the tilt shape so the identity is never overspent, then
    hands single units back to the bands with the largest lost fraction while
    they still fit.  The leftover is bounded by max(w_b) / (R * L_k), which is
    far inside the loader's underspend tolerance -- an exact integer solution
    does not exist for arbitrary widths, and overspending to reach one would
    price the cell as cheaper than it runs.
    """
    n_bands = len(widths)
    total_width = sum(widths)
    # Straight line from (1 + tilt) at band 0 down to (1 - tilt) at the last.
    if n_bands == 1:
        shape = [1.0]
    else:
        shape = [1.0 + tilt - 2.0 * tilt * b / (n_bands - 1)
                 for b in range(n_bands)]
    # Renormalize so the shape itself is width-weighted iso-compute; otherwise
    # unequal band widths bias the mean and every rung starts overspent.
    mean = sum(widths[b] * shape[b] for b in range(n_bands)) / total_width
    shape = [s / mean for s in shape]

    ladder = []
    for parent in parent_levels:
        if parent == 0:
            ladder.append([0] * n_bands)
            continue
        budget = total_width * parent            # sum_b w_b * L[b] must be <=
        raw = [parent * s for s in shape]
        lengths = []
        for b in range(n_bands):
            v = int(raw[b] // snap) * snap
            v = max(snap, min(v, max_stoc_len))
            lengths.append(v)
        spent = sum(widths[b] * lengths[b] for b in range(n_bands))
        if spent > budget:
            # Only reachable via the max_stoc_len clamp lifting a band.
            order = sorted(range(n_bands), key=lambda b: -lengths[b])
            for b in order:
                while spent > budget and lengths[b] > snap:
                    lengths[b] -= snap
                    spent -= widths[b] * snap
        # Hand back what is left, largest lost fraction first.
        order = sorted(range(n_bands), key=lambda b: -(raw[b] - lengths[b]))
        progressed = True
        while progressed:
            progressed = False
            for b in order:
                if lengths[b] + snap > max_stoc_len:
                    continue
                if spent + widths[b] * snap <= budget:
                    lengths[b] += snap
                    spent += widths[b] * snap
                    progressed = True
        ladder.append(lengths)

    # ladder is [n_rungs][n_bands]; the table stores [n_bands][n_rungs].
    return [[ladder[k][b] for k in range(len(parent_levels))]
            for b in range(n_bands)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", required=True,
                   help="existing adaptive MP table (JSON)")
    p.add_argument("--out", dest="dst", required=True)
    p.add_argument("--n-bands", type=int, default=4,
                   help="maximum bands; clamped per operator by its chunk count")
    p.add_argument("--chunk-d", type=int, default=128)
    p.add_argument("--tilt", type=float, default=0.0,
                   help="0 reproduces the per-row parent exactly; 0.25 runs "
                        "the first band ~25%% longer and the last ~25%% shorter")
    p.add_argument("--max-stoc-len", type=int, default=0,
                   help="halve-mode ceiling, e.g. 128 for sc_prec=8 (0 = none)")
    p.add_argument("--snap", type=int, default=1,
                   help="round stream lengths to a multiple of this")
    p.add_argument("--operators", default="mlp_fc1,mlp_fc2,proj,input_proj",
                   help="comma-separated operators to band")
    p.add_argument("--blocks", type=int, default=28,
                   help="number of transformer blocks (28 for DiT-XL/2); the "
                        "band map is emitted per (operator, block)")
    p.add_argument("--width", action="append", default=[], metavar="OP=N",
                   help="override an operator's contraction width")
    args = p.parse_args()

    with open(args.src) as f:
        payload = json.load(f)

    levels = [int(x) for x in payload["stoc_len_levels"]]
    max_stoc_len = args.max_stoc_len or max(levels)
    widths_by_op = dict(DEFAULT_WIDTHS)
    for spec in args.width:
        op, _, n = spec.partition("=")
        widths_by_op[op] = int(n)

    operators = [op for op in args.operators.split(",") if op]
    # Only band operators the table actually has thresholds for; a band map for
    # an operator the run never dispatches is dead weight that still validates.
    seen_ops = {key.split(":")[0] for key in payload.get("buckets", {})}
    seen_ops |= set(payload.get("operator_defaults", {}))

    n_blocks = args.blocks
    residual_width, chunk_bands, ladders = {}, {}, {}
    for op in operators:
        if op not in seen_ops:
            print(f"[build_kband_table] skipping '{op}': no thresholds in the "
                  f"source table", file=sys.stderr)
            continue
        if op not in widths_by_op:
            raise SystemExit(
                f"no contraction width known for '{op}'; pass --width {op}=N")
        R = widths_by_op[op]
        cw = chunk_widths(R, args.chunk_d)
        n_bands = min(args.n_bands, len(cw) // 2)
        if n_bands < 2:
            print(f"[build_kband_table] skipping '{op}': {len(cw)} chunks at "
                  f"chunk_d={args.chunk_d} leaves room for {len(cw)//2} bands",
                  file=sys.stderr)
            continue
        if n_bands < args.n_bands:
            print(f"[build_kband_table] '{op}': capped to {n_bands} bands "
                  f"({len(cw)} chunks)", file=sys.stderr)
        bands = contiguous_bands(len(cw), n_bands)
        bw = [0] * n_bands
        for c, b in enumerate(bands):
            bw[b] += cw[c]

        residual_width[op] = R
        for blk in range(n_blocks):
            chunk_bands[f"{op}:{blk}"] = bands
        band_ladder = solve_ladder(levels, bw, args.tilt, max_stoc_len,
                                   args.snap)
        for t in range(int(payload.get("timestep_buckets", 1))):
            for l in range(int(payload.get("layer_buckets", 1))):
                ladders[f"{op}:t{t}:l{l}"] = band_ladder

        spend = [sum(bw[b] * band_ladder[b][k] for b in range(n_bands)) /
                 (sum(bw) * levels[k]) if levels[k] else 1.0
                 for k in range(len(levels))]
        print(f"[build_kband_table] {op}: {n_bands} bands, widths {bw}, "
              f"budget used per rung {[f'{s:.4f}' for s in spend]}")

    if not residual_width:
        raise SystemExit("no operator could be banded; nothing written")

    # n_bands in the section is the max across operators; the loader checks
    # band ids against it and the per-operator ladder length carries the rest.
    payload["k_bands"] = {
        "n_bands": max(len(v) for v in ladders.values()),
        "chunk_d": args.chunk_d,
        "residual_width": residual_width,
        "chunk_bands": chunk_bands,
        "ladders": ladders,
    }
    with open(args.dst, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[build_kband_table] wrote {args.dst}")


if __name__ == "__main__":
    main()
