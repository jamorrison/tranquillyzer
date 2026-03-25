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
        _plot_read_length_dist,
        _plot_read_length_per_cell,
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

    # ── collect plot rows ────────────────────────────────────────────────────
    # Each row is a list of (title, fig, caption).
    # Two-item rows → side by side; single-item → full width (colspan=2).
    rows = []

    arch = _plot_read_architecture(summary, sample_name)
    bc   = _plot_barcode_assignment(summary, sample_name)
    rows.append([arch, bc] if bc is not None else [arch])

    # Invalid read reasons (full-width row, before edit distances).
    inv_reasons = _plot_invalid_reasons(invalid_path)
    if inv_reasons is not None:
        rows.append([inv_reasons])

    # Edit-distance row: all barcode types in a single row.
    ed_items = _plot_edit_distances(valid_path, barcode_types, vcols)
    if ed_items:
        rows.append(ed_items)

    # Knee plot (full-width row, before read-length distribution).
    knee = _plot_knee(valid_path, vcols)
    if knee is not None:
        rows.append([knee])

    # Read-length distribution (full-width row).
    rl = _plot_read_length_dist(valid_path, invalid_path, vcols, read_len_bin_width)
    if rl is not None:
        rows.append([rl])

    # Per-cell read-length summary (full-width row).
    rl_cell = _plot_read_length_per_cell(valid_path, vcols)
    if rl_cell is not None:
        rows.append([rl_cell])

    # Future rows appended here:
    # rows.append([_plot_knee(...)])

    # ── build per-row figures and write HTML report ───────────────────────────
    n_cols = max(max(len(row) for row in rows), 2)
    row_figs = [_build_row_figure(row, n_cols) for row in rows]
    report_path = os.path.join(output_dir, f"{sample_name}_qc_report.html")
    _write_html_report(report_path, row_figs, sample_name)
    logger.info(f"QC report complete -> {report_path}")
