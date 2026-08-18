"""Wrap sc_matmul to count calls per granularity, then run quant_sc_main.py.

Confirms SC kernels are actually invoked during DiT sampling, not skipped.
Prints a count summary at process exit.
"""
import sys, os, atexit
from pathlib import Path
from collections import Counter

ROOT = Path('/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_diffusion')
sys.path.insert(0, str(ROOT))

import scmp_kernels.sc.matmul as mm
_orig = mm.sc_matmul
counts = Counter()

def _wrapped(a, b, granularity="per_row", **kw):
    key = (
        granularity,
        kw.get('mode', 'bipolar'),
        'chunk' if kw.get('chunk_d', 0) > 0 else 'nochunk',
        'group' if (kw.get('group_a', 1) > 1 or kw.get('group_b', 1) > 1) else 'nogroup',
    )
    counts[key] += 1
    return _orig(a, b, granularity=granularity, **kw)

mm.sc_matmul = _wrapped

# Patch every site that did `from scmp_kernels.sc import sc_matmul`
import scmp_kernels.sc as sc_pkg
sc_pkg.sc_matmul = _wrapped
import qdit.sc_integration.sc_attention as sa
import qdit.sc_integration.sc_mlp as sm
sa.sc_matmul = _wrapped
sm.sc_matmul = _wrapped

def _summary():
    total = sum(counts.values())
    print(f"\n========== sc_matmul call summary ==========", flush=True)
    print(f"Total sc_matmul invocations: {total}", flush=True)
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        gran, mode, chunk, group = k
        print(f"  {n:>6}  granularity={gran:<10} mode={mode:<8} {chunk:<8} {group}", flush=True)
    print(f"============================================\n", flush=True)
atexit.register(_summary)

# Now run quant_sc_main as if invoked directly
sys.argv = [
    'quant_sc_main.py',
    '--ckpt', '/nfs/turbo/coe-nbleier/zhkangqi/pretrained_models/DiT-XL-2-256x256.pt',
    '--wbits', '8', '--abits', '8', '--w_sym', '--a_sym',
    '--timewise', '0.5', '--qklayerwise', '0.8',
    '--avlayerwise', '0.0', '--projlayerwise', '0.0',
    '--mlplayerwise', '0.0', '--inputprojlayerwise', '0.0',
    '--sc_prec', '8',
    '--image-size', '256', '--num-sampling-steps', '50',
    '--cfg-scale', '4', '--batch-size', '8',
    '--results-dir', 'results/smoke_e2e_count',
]
os.chdir(str(ROOT))
exec(compile(open('scripts/quant_sc_main.py').read(), 'scripts/quant_sc_main.py', 'exec'))
