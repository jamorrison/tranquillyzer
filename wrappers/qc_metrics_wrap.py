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
):
    """Generate QC metrics report from annotation outputs."""
    import os

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

    def _match_yaxis_range(figs):
        """Set the same y-axis range across figures (default-visible traces only)."""
        y_max = 0
        for fig in figs:
            for trace in fig.data:
                if trace.visible is not False and hasattr(trace, "y") and trace.y is not None:
                    vals = [v for v in trace.y if v is not None]
                    if vals:
                        y_max = max(y_max, max(vals))
        if y_max > 0:
            for fig in figs:
                fig.update_yaxes(range=[0, y_max * 1.05])

    # ── collect plot rows ────────────────────────────────────────────────────
    # Each row is a list of (title, fig, caption).
    # Multi-item rows → side by side; single-item → full width (colspan=2).
    rows = []

    # 1. Read Architecture.
    arch = _plot_read_architecture(summary, sample_name)
    rows.append([arch])

    # 2. Invalid read reasons.
    inv_reasons = _plot_invalid_reasons(invalid_path)
    if inv_reasons is not None:
        rows.append([inv_reasons])

    # 3. Barcode Discovery + Reads per Cell (side by side when both present).
    metadata_dir = os.path.join(input_dir, "annotation_metadata")
    knee_wl_free = _plot_knee_whitelist_free(metadata_dir)
    knee = _plot_knee(valid_path, vcols)
    if knee_wl_free is not None and knee is not None:
        rows.append([knee_wl_free, knee])
    elif knee_wl_free is not None:
        rows.append([knee_wl_free])
    elif knee is not None:
        rows.append([knee])

    # 4-5. Edit-distance + Barcode Assignment.
    #       Side by side when there is exactly one edit-distance plot;
    #       otherwise separate rows with captions beside the plots.
    ed_items = _plot_edit_distances(valid_path, barcode_types, vcols)
    bc = _plot_barcode_assignment(summary, sample_name)
    if ed_items and len(ed_items) == 1 and bc is not None:
        rows.append([ed_items[0], bc])
    else:
        if ed_items:
            rows.append([(*it, "side") if len(it) <= 4 else it
                         for it in ed_items])
        if bc is not None:
            bc_side = (*bc, "side") if len(bc) <= 4 else bc
            rows.append([bc_side])

    # 6. Read-length distribution.
    rl = _plot_read_length_dist(valid_path, invalid_path, vcols, read_len_bin_width)
    if rl is not None:
        rows.append([rl])

    # 7. cDNA-length distribution.
    cdna_rl = _plot_cdna_length_dist(valid_path, vcols, read_len_bin_width)
    if cdna_rl is not None:
        rows.append([cdna_rl])

    # 8. Read length per cell + cDNA length per cell (side by side).
    rl_cell = _plot_read_length_per_cell(valid_path, vcols)
    cdna_cell = _plot_cdna_length_per_cell(valid_path, vcols)
    if rl_cell is not None and cdna_cell is not None:
        rows.append([rl_cell, cdna_cell])
    elif rl_cell is not None:
        rows.append([rl_cell])
    elif cdna_cell is not None:
        rows.append([cdna_cell])

    # ── build per-row figures and write HTML report ───────────────────────────
    n_cols = max(max(len(row) for row in rows), 2)
    row_figs = []
    for row in rows:
        shared_y = len(row) > 1  # side-by-side rows share y-axis
        row_figs.append(_build_row_figure(row, n_cols, shared_yaxes=shared_y))
    report_path = os.path.join(output_dir, f"{sample_name}_qc_report.html")
    _write_html_report(report_path, row_figs, sample_name)
    logger.info(f"QC report complete -> {report_path}")
