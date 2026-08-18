"""K-bands: per-group (contraction-axis) stream lengths.

Row dispatch is unchanged -- one metric, one rung index per row -- but the
contraction axis is partitioned into bands of whole quantization chunks and
band ``b`` runs rung ``k`` at its own stream length ``L[b][k]``.

Why this cannot lose: setting every band's ladder equal to the parent ladder
(``L[b][k] == stoc_len_levels[k]``) reproduces the per-row parent, so the
parent sits inside the search space.  The budget is an exact per-rung identity
rather than a tolerance, because MACs are linear in the contraction dim and a
band's column fraction therefore IS its MAC fraction::

    sum_b (w_b / R) * L[b][k] == L_parent[k]

Why whole chunks: the kernel builds its RNG tables over ``chunk_d`` dims and
reuses them for every chunk, so a chunk relocated to another band keeps the
exact scale and Owen mask it would have had in the unbanded call.  Relocating
arbitrary channels does not.

The allocation itself (which chunk goes in which band, and each band's ladder)
is solved offline and shipped in the MP table's ``k_bands`` section; see
``AdaptiveMPConfig._load_k_bands`` for the schema and its validation.
"""

import torch


def resolve_k_bands(sc_controller, operator, block_idx, D, chunk_d):
    """Band map + ladders for one module, or ``None`` to run the per-row parent.

    Returns ``(band_of_chunk, band_ladders)``.  ``None`` is returned whenever
    the adaptive config carries no allocation for this module, which is the
    normal path for operators the solver left alone.

    A table that DOES cover this module but was priced against a different
    contraction width or chunk size raises instead of falling back: the band
    map would no longer describe this matmul, and silently running the parent
    while the run is labelled "k-bands" is the one failure mode that looks
    exactly like a result.
    """
    mp_config = getattr(sc_controller, "adaptive_mp_config", None)
    if mp_config is None or not getattr(mp_config, "k_band_count", 0):
        return None

    k_bands = mp_config.get_k_bands(
        operator, block_idx, sc_controller.total_blocks,
        timestep=sc_controller.current_timestep,
        total_timesteps=sc_controller.total_timesteps,
    )
    if k_bands is None:
        return None

    table_chunk_d = int(mp_config.k_band_chunk_d)
    if chunk_d != table_chunk_d:
        raise ValueError(
            f"K-band table for {operator} block {block_idx} was solved at "
            f"chunk_d={table_chunk_d} but this call runs chunk_d={chunk_d}. "
            f"Bands own whole quantization chunks, so the band map does not "
            f"describe this matmul.")
    table_width = int(mp_config.k_band_residual_width[operator])
    if D != table_width:
        raise ValueError(
            f"K-band table for {operator} block {block_idx} was priced against "
            f"a contraction width of {table_width} but this call has D={D}. "
            f"Band widths set the iso-compute budget, so the allocation is "
            f"not transferable.")
    if D <= chunk_d:
        raise ValueError(
            f"K-band table covers {operator} block {block_idx} but D={D} is "
            f"within a single chunk of {chunk_d}; there is nothing to band.")
    return k_bands


def band_columns(module, band_of_chunk, n_bands, chunk_d, D, device):
    """Column indices owned by each band, plus each band's channel width.

    Chunks stay in ASCENDING order inside a band, which puts the short tail
    chunk last in whichever band owns it.  That is the only ordering under
    which a band's gathered columns re-chunk to the same boundaries the
    unbanded call would have used, and therefore the only one under which the
    per-chunk scales and RNG tables are unchanged.

    Cached on the module: the band map is fixed for a module's lifetime, while
    this runs per forward.
    """
    cache = getattr(module, "_sc_k_band_cache", None)
    key = (tuple(band_of_chunk), n_bands, chunk_d, D, str(device))
    if cache is not None and cache[0] == key:
        return cache[1], cache[2]

    cols_per_band, width_per_band = [], []
    for b in range(n_bands):
        cols = []
        for chunk_idx, band in enumerate(band_of_chunk):
            if band != b:
                continue
            start = chunk_idx * chunk_d
            cols.extend(range(start, min(start + chunk_d, D)))
        width_per_band.append(len(cols))
        cols_per_band.append(
            torch.tensor(cols, dtype=torch.long, device=device))

    if sum(width_per_band) != D:
        raise ValueError(
            f"K-band columns cover {sum(width_per_band)} of {D} channels; the "
            f"band map does not partition the contraction axis.")

    module._sc_k_band_cache = (key, cols_per_band, width_per_band)
    return cols_per_band, width_per_band


def _reject_k_bands_on_combined_mp(sc_controller, operator, block_idx, D,
                                   chunk_d):
    """Refuse to run the combined range+dynamic path on a banded operator.

    That path iterates range-based weight groups and takes
    ``min(range_stoc_len, dynamic_stoc_len)`` per (group, row).  It has no
    contraction-axis dispatch, so a band allocation for this operator would be
    dropped on the floor while the run still carries the k-bands label.
    """
    mp_config = getattr(sc_controller, "adaptive_mp_config", None)
    if mp_config is None or not getattr(mp_config, "k_band_count", 0):
        return
    if resolve_k_bands(sc_controller, operator, block_idx, D, chunk_d) is None:
        return
    raise NotImplementedError(
        f"K-bands and range-based per-group weight dispatch are both active "
        f"for '{operator}' (block {block_idx}). The combined path has no "
        f"contraction-axis dispatch and would silently discard the band "
        f"allocation. Drop --range_mp, or drop '{operator}' from the table's "
        f"k_bands section.")


def band_of_chunk_index(band_of_chunk, chunk_idx):
    """Band owning chunk ``chunk_idx``, bounds-checked.

    Used by the app-level chunk loops (attention), which walk chunks directly
    instead of gathering per-band columns.
    """
    if not 0 <= chunk_idx < len(band_of_chunk):
        raise ValueError(
            f"chunk index {chunk_idx} outside the band map of length "
            f"{len(band_of_chunk)}; the table's chunk count and the runtime "
            f"chunking disagree.")
    return band_of_chunk[chunk_idx]
