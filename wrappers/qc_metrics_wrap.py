"""
QC metrics from tranquillyzer annotation parquet files.

Outputs a self-contained Plotly HTML report.  Plots are added incrementally:
each ``_plot_*`` function returns a ``(section_title, go.Figure, caption)`` tuple;
the orchestrator groups them into rows, builds one ``make_subplots`` figure per row
(each with its own independent modebar), then combines all into one HTML file.

Design:
  - The parquet schema is probed once (zero data loaded).
  - Each metric/plot function receives file paths and loads only the columns it
    needs via ``pl.scan_parquet(...).select([...]).collect()``.
  - Independent metric functions run in parallel via ThreadPoolExecutor
    (Polars releases the GIL, giving real I/O concurrency).
  - ``qc_metrics_wrap`` is a thin orchestrator that collects figures and writes
    the single HTML report.
"""

import logging

logger = logging.getLogger(__name__)


def qc_metrics_wrap(
    input_dir,
    output_dir,
    valid_file,
    invalid_file,
    sample_name,
    read_len_bin_width,
    bam_file=None,
    threads=4,
):
    """Generate QC metrics report from annotation outputs."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    from scripts.qc_metrics import (
        _find_file,
        _probe_schema,
        _detect_barcode_types,
        _compute_summary,
        _plot_read_architecture,
        _plot_barcode_assignment,
        _plot_invalid_reasons,
        _plot_edit_distances,
        _plot_knee,
        _plot_knee_whitelist_free,
        _plot_read_length_dist,
        _plot_cdna_length_dist,
        _plot_read_length_per_cell,
        _plot_cdna_length_per_cell,
        _plot_segment_lengths,
        _collect_bam_per_cell_stats,
        _plot_saturation_curve,
        _plot_alignment_stats,
        _plot_global_dup_stats,
        _plot_mapping_rate_per_cell,
        _plot_dup_rate_per_cell,
        _build_row_figure,
        _write_html_report,
    )

    os.makedirs(output_dir, exist_ok=True)

    # ── resolve file paths ───────────────────────────────────────────────────
    metadata_dir = os.path.join(input_dir, "annotation_metadata")
    valid_path = _find_file(input_dir, [valid_file] if valid_file else []) or _find_file(
        metadata_dir, ["annotations_valid_bc_corrected.parquet", "annotations_valid.parquet"]
    )
    invalid_path = _find_file(input_dir, [invalid_file] if invalid_file else []) or _find_file(
        metadata_dir, ["annotations_invalid.parquet"]
    )

    if valid_path is None and invalid_path is None:
        raise FileNotFoundError(
            f"No annotation parquet files found in '{input_dir}'. "
            "Expected annotations_valid*.parquet / annotations_invalid.parquet."
        )

    logger.info(f"Valid   parquet : {valid_path}")
    logger.info(f"Invalid parquet : {invalid_path}")

    # ── probe schemas (zero data loaded) ────────────────────────────────────
    vcols         = _probe_schema(valid_path)
    barcode_types = _detect_barcode_types(list(vcols))

    # ── compute shared summary data ──────────────────────────────────────────
    summary = _compute_summary(valid_path, invalid_path, vcols)

    # ── run independent metric functions in parallel ─────────────────────────
    with ThreadPoolExecutor(max_workers=threads) as pool:
        f_arch      = pool.submit(_plot_read_architecture, summary, sample_name)
        f_seg       = pool.submit(_plot_segment_lengths, valid_path, vcols)
        f_inv       = pool.submit(_plot_invalid_reasons, invalid_path)
        f_knee_wl   = pool.submit(_plot_knee_whitelist_free, metadata_dir)
        f_knee      = pool.submit(_plot_knee, valid_path, vcols)
        f_ed        = pool.submit(_plot_edit_distances, valid_path, barcode_types, vcols)
        f_bc        = pool.submit(_plot_barcode_assignment, summary, sample_name)
        f_rl        = pool.submit(_plot_read_length_dist, valid_path, invalid_path, vcols, read_len_bin_width)
        f_cdna_rl   = pool.submit(_plot_cdna_length_dist, valid_path, vcols, read_len_bin_width)
        f_rl_cell   = pool.submit(_plot_read_length_per_cell, valid_path, vcols)
        f_cdna_cell = pool.submit(_plot_cdna_length_per_cell, valid_path, vcols)
        f_bam       = pool.submit(_collect_bam_per_cell_stats, bam_file, threads) if bam_file is not None else None

        # Collect parquet metric results
        def _collect(name, fut):
            logger.info(f"  Collecting {name} ...")
            r = fut.result()
            logger.info(f"  Collected  {name}")
            return r

        arch        = _collect("read_architecture", f_arch)
        seg_all, seg_demux = _collect("segment_lengths", f_seg)
        inv_reasons = _collect("invalid_reasons", f_inv)
        knee_wl_free = _collect("knee_whitelist_free", f_knee_wl)
        knee        = _collect("knee", f_knee)
        ed_items    = _collect("edit_distances", f_ed)
        bc          = _collect("barcode_assignment", f_bc)
        rl          = _collect("read_length_dist", f_rl)
        cdna_rl     = _collect("cdna_length_dist", f_cdna_rl)
        rl_cell     = _collect("read_length_per_cell", f_rl_cell)
        cdna_cell   = _collect("cdna_length_per_cell", f_cdna_cell)

        # Collect BAM scan result and run BAM-dependent plots in parallel
        bam_stats_df = None
        umi_pairs_df = None
        has_dedup = False
        if f_bam is not None:
            bam_stats_df, umi_pairs_df, has_dedup = f_bam.result()

        logger.info("Computing BAM-derived plots...")
        f_sat     = pool.submit(_plot_saturation_curve, umi_pairs_df) if umi_pairs_df is not None else None
        f_aln     = pool.submit(_plot_alignment_stats, bam_stats_df) if bam_stats_df is not None else None
        f_dup_g   = pool.submit(_plot_global_dup_stats, bam_file) if bam_file is not None else None
        f_mapping = pool.submit(_plot_mapping_rate_per_cell, bam_stats_df) if bam_stats_df is not None else None
        f_dup_c   = pool.submit(_plot_dup_rate_per_cell, bam_stats_df, has_dedup) if bam_stats_df is not None else None

        sat       = f_sat.result() if f_sat is not None else None
        aln_stats = f_aln.result() if f_aln is not None else None
        dup_stats = f_dup_g.result() if f_dup_g is not None else None
        mapping   = f_mapping.result() if f_mapping is not None else None
        dup       = f_dup_c.result() if f_dup_c is not None else None

    # ── assemble rows ────────────────────────────────────────────────────────
    rows = []

    # 1. Read Architecture.
    rows.append([arch])

    # 2. Segment lengths (all valid reads).
    if seg_all is not None:
        rows.append([seg_all])

    # 3. Invalid read reasons.
    if inv_reasons is not None:
        rows.append([inv_reasons])

    # 4. Barcode Discovery + Reads per Cell (side by side when both present).
    if knee_wl_free is not None and knee is not None:
        rows.append([knee_wl_free, knee])
    elif knee_wl_free is not None:
        rows.append([knee_wl_free])
    elif knee is not None:
        rows.append([knee])

    # 5-6. Edit-distance + Barcode Assignment.
    if ed_items and len(ed_items) == 1 and bc is not None:
        rows.append([ed_items[0], bc])
    else:
        if ed_items:
            rows.append([(*it, "side") if len(it) <= 4 else it
                         for it in ed_items])
        if bc is not None:
            bc_side = (*bc, "side") if len(bc) <= 4 else bc
            rows.append([bc_side])

    # 7. Segment lengths (demuxed reads only).
    if seg_demux is not None:
        rows.append([seg_demux])

    # 8. Read-length distribution.
    if rl is not None:
        rows.append([rl])

    # 9. cDNA-length distribution.
    if cdna_rl is not None:
        rows.append([cdna_rl])

    # 10. Read length per cell + cDNA length per cell (side by side).
    if rl_cell is not None and cdna_cell is not None:
        rows.append([rl_cell, cdna_cell])
    elif rl_cell is not None:
        rows.append([rl_cell])
    elif cdna_cell is not None:
        rows.append([cdna_cell])

    # 11. Saturation curve.
    if sat is not None:
        rows.append([sat])

    # 12. Alignment stats + Duplication stats (side by side).
    if aln_stats is not None and dup_stats is not None:
        rows.append([aln_stats, dup_stats])
    elif aln_stats is not None:
        rows.append([aln_stats])
    elif dup_stats is not None:
        rows.append([dup_stats])

    # 13. Mapping rate per cell + Dup rate per cell (side by side).
    if mapping is not None and dup is not None:
        rows.append([mapping, dup])
    elif mapping is not None:
        rows.append([mapping])
    elif dup is not None:
        rows.append([dup])

    # ── build per-row figures and write HTML report ───────────────────────────
    logger.info("Assembling HTML report...")
    n_cols = max(max(len(row) for row in rows), 2)
    row_figs = []
    for row in rows:
        shared_y = len(row) > 1  # side-by-side rows share y-axis
        row_figs.append(_build_row_figure(row, n_cols, shared_yaxes=shared_y))
    report_path = os.path.join(output_dir, f"{sample_name}_qc_report.html")
    _write_html_report(report_path, row_figs, sample_name)
    logger.info(f"QC report complete -> {report_path}")
