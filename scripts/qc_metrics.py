"""
Helper functions for the QC metrics HTML report.

I/O utilities, schema helpers, Polars filter expressions,
Plotly figure builders, and individual metric/plot functions.
All functions are verbatim from qc_metrics_wrap.py — only moved here.
"""

import logging
import os
import re

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────


def _find_file(input_dir, candidates):
    """Return the first existing candidate path (absolute or relative to input_dir)."""
    for c in candidates:
        if not c:
            continue
        p = c if os.path.isabs(c) else os.path.join(input_dir, c)
        if os.path.exists(p):
            return p
    return None


def _probe_schema(path):
    """Return the set of column names without loading any data."""
    if path is None:
        return set()
    return set(pl.read_parquet_schema(path).keys())


def _scan_cols(path, cols, schema=None):
    """
    Lazy-load only the requested columns from a parquet file and collect.
    Columns absent from the file are silently skipped.

    When *schema* is provided (a set of column names), it is used instead
    of probing the parquet file — avoiding a redundant file open.
    """
    available = schema if schema is not None else _probe_schema(path)
    wanted = [c for c in cols if c in available]
    if not wanted:
        return pl.DataFrame()
    return pl.scan_parquet(path).select(wanted).collect()


def _count_rows(path):
    """Return row count without loading column data."""
    if path is None:
        return 0
    return pl.scan_parquet(path).select(pl.len()).collect()[0, 0]


# ─────────────────────────────────────────────────────────────────────────────
# schema helpers
# ─────────────────────────────────────────────────────────────────────────────


_BC_MIN_DIST_RE = re.compile(r"^corrected_(.+)_min_dist$")


def _detect_barcode_types(col_names):
    """
    Return barcode names that have a ``corrected_{bc}_min_dist`` column.
    Guards against false matches like ``corrected_CBC_counts_with_min_dist``.
    """
    types = []
    for col in col_names:
        m = _BC_MIN_DIST_RE.match(col)
        if m and not m.group(1).endswith("_counts_with"):
            types.append(m.group(1))
    return types


def _first_present(col_names, candidates):
    """Return the first candidate that appears in col_names, or None."""
    for c in candidates:
        if c in col_names:
            return c
    return None


# ── shared cell_id expressions ────────────────────────────────────────────────
# cell_id is numeric (as string) → demuxed; "ambiguous" string → ambiguous.

def _expr_is_demuxed(cell_col):
    """Polars expression: read has a resolved numeric cell assignment."""
    col = pl.col(cell_col).cast(pl.Utf8)
    return (
        pl.col(cell_col).is_not_null()
        & (col != "ambiguous")
        & (col != "")
    )


def _expr_is_ambiguous(cell_col):
    """Polars expression: read has the 'ambiguous' sentinel cell_id."""
    return pl.col(cell_col).cast(pl.Utf8) == "ambiguous"


def _add_cdna_length(df):
    """Add ``cDNA_length`` column from scalar ``cDNA_Starts`` / ``cDNA_Ends``."""
    return df.with_columns(
        (pl.col("cDNA_Ends").cast(pl.Int64) - pl.col("cDNA_Starts").cast(pl.Int64))
        .alias("cDNA_length")
    )


# ─────────────────────────────────────────────────────────────────────────────
# report builder  (pure Plotly — no HTML/CSS)
# ─────────────────────────────────────────────────────────────────────────────


_TAG_RE = re.compile(r"<[^>]+>|&[a-z]+;")


def _wrap_caption(text, col_span, n_cols, full_chars=150):
    """
    Word-wrap ``text`` to fit within the fraction ``col_span / n_cols`` of the
    figure width.  Paragraphs are separated by ``\\n\\n`` in the source and
    joined with ``<br>`` in the output; HTML tags and entities are excluded
    from character-count to avoid premature breaks.
    """
    chars = max(int(full_chars * col_span / n_cols), 35)
    parts = []
    for para in text.split("\n\n"):
        words = para.split()
        lines, buf, buf_len = [], [], 0
        for word in words:
            wlen = len(_TAG_RE.sub("", word))
            space = 1 if buf else 0
            if buf_len + space + wlen > chars and buf:
                lines.append(" ".join(buf))
                buf, buf_len = [word], wlen
            else:
                buf.append(word)
                buf_len += space + wlen
        if buf:
            lines.append(" ".join(buf))
        parts.append("<br>".join(lines))
    return "<br>".join(parts)


def _row_item(item):
    """Unpack a row item as (title, fig, caption, caption_y, caption_xalign).

    Accepts 3-tuples ``(title, fig, caption)`` — caption_y defaults to -0.12,
    caption_xalign to ``"center"`` — 4-tuples ``(title, fig, caption, caption_y)``
    or 5-tuples ``(title, fig, caption, caption_y, caption_xalign)``.
    """
    if len(item) == 5:
        return item
    if len(item) == 4:
        return (*item, "center")
    return (*item, -0.12, "center")


def _build_row_figure(row, n_cols, shared_yaxes=False):
    """
    Build a ``go.Figure`` for a single row of subplots using ``make_subplots``.
    Each row figure has its own Plotly modebar with independent zoom/pan/autoscale.

    Parameters
    ----------
    row : list of ``(title, go.Figure, caption[, caption_y])`` tuples
    n_cols : int
        Global column count so caption word-wrap matches across rows.
    shared_yaxes : bool
        If True, subplots in the row share the same y-axis scale.

    Returns
    -------
    go.Figure
    """
    k = len(row)

    if k == n_cols:
        spec_row   = [{} for _ in range(n_cols)]
        colspans   = [1] * k
        start_cols = list(range(1, n_cols + 1))
    elif k == 1:
        spec_row   = [{"colspan": n_cols}] + [None] * (n_cols - 1)
        colspans   = [n_cols]
        start_cols = [1]
    else:
        base_span  = n_cols // k
        extra      = n_cols % k
        spec_row, start_cols, colspans = [], [], []
        col_cursor = 1
        for i in range(k):
            span = base_span + (1 if i < extra else 0)
            spec_row.append({"colspan": span} if span > 1 else {})
            spec_row.extend([None] * (span - 1))
            start_cols.append(col_cursor)
            colspans.append(span)
            col_cursor += span

    subplot_titles = [_row_item(item)[0] for item in row]

    combined = make_subplots(
        rows=1, cols=n_cols,
        specs=[spec_row],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        shared_yaxes=shared_yaxes,
    )

    # Style subplot titles — bold via <b> tag.
    for ann in combined.layout.annotations:
        if ann.y is not None and ann.y >= 1.0:
            ann.text = f"<b>{ann.text}</b>"
            ann.font = dict(size=17, color="#1a1a2e", family="Arial")

    subplot_meta = []  # (col_i, trace_start, trace_count, fig)

    for col_i, item in enumerate(row):
        _, fig, caption, caption_y, caption_xalign = _row_item(item)
        c   = start_cols[col_i]
        sfx = "" if col_i == 0 else str(col_i + 1)

        trace_start = len(combined.data)
        for trace in fig.data:
            combined.add_trace(trace, row=1, col=c)
        trace_count = len(combined.data) - trace_start

        subplot_meta.append((col_i, trace_start, trace_count, fig))

        combined.update_xaxes(fig.layout.xaxis.to_plotly_json(), row=1, col=c)
        combined.update_yaxes(fig.layout.yaxis.to_plotly_json(), row=1, col=c)

        if caption and caption_xalign == "side":
            # Place caption to the right of the plot area.
            ax_key = "xaxis" if sfx == "" else f"xaxis{sfx}"
            ay_key = "yaxis" if sfx == "" else f"yaxis{sfx}"
            xdom = list(combined.layout[ax_key].domain)
            ydom = list(combined.layout[ay_key].domain)
            # Shrink plot to ~65% of its width; caption fills the rest.
            orig_width = xdom[1] - xdom[0]
            orig_center = (xdom[0] + xdom[1]) / 2
            plot_frac = 0.65
            new_right = xdom[0] + orig_width * plot_frac
            combined.layout[ax_key].domain = [xdom[0], new_right]
            # Re-center the subplot title over the shrunk plot area.
            new_center = (xdom[0] + new_right) / 2
            for ann in combined.layout.annotations:
                if abs(ann.x - orig_center) < 0.01 and ann.y >= 1.0:
                    ann.x = new_center
                    break
            cap_x = new_right + 0.01
            cap_y = (ydom[0] + ydom[1]) / 2
            combined.add_annotation(
                xref="paper", yref="paper",
                x=cap_x, y=cap_y,
                xanchor="left", yanchor="middle",
                text=_wrap_caption(caption, colspans[col_i], n_cols,
                                   full_chars=45),
                showarrow=False,
                align="left",
                font=dict(size=11, color="#666"),
            )
        elif caption:
            combined.add_annotation(
                xref=f"x{sfx} domain", yref=f"y{sfx} domain",
                x=0.5, y=caption_y,
                xanchor="center", yanchor="top",
                text=_wrap_caption(caption, colspans[col_i], n_cols),
                showarrow=False,
                align="center",
                font=dict(size=12, color="#666"),
            )

    # Bar traces never need a legend entry; scatter/line traces do.
    combined.update_traces(showlegend=False, selector=dict(type="bar"))

    # Use unified hover for line-chart rows (no bar traces present).
    has_bars = any(t.type == "bar" for t in combined.data)
    has_boxes = any(t.type == "box" for t in combined.data)
    if not has_bars and not has_boxes:
        combined.update_layout(hovermode="x unified")

    # Propagate updatemenus and extra annotations from individual figures.
    # Trace-index arrays are padded so each subplot's buttons only toggle its
    # own traces, and menu/annotation x-positions are mapped into the
    # subplot's x-domain so they don't overlap when two plots share a row.
    total_traces = len(combined.data)
    extra_menus = []
    extra_annots = []

    for col_i, trace_start, trace_count, fig in subplot_meta:
        sfx = "" if col_i == 0 else str(col_i + 1)
        ax_key = "xaxis" if sfx == "" else f"xaxis{sfx}"
        xdom = list(combined.layout[ax_key].domain)
        dom_width = xdom[1] - xdom[0]

        if fig.layout.updatemenus:
            for um in fig.layout.updatemenus:
                um_dict = um.to_plotly_json()
                # Reposition x into this subplot's domain.
                local_x = um_dict.get("x", 0)
                um_dict["x"] = xdom[0] + local_x * dom_width
                # Pad visible/showlegend arrays to address correct traces.
                for btn in um_dict.get("buttons", []):
                    args_list = btn.get("args", [{}])
                    if not args_list:
                        continue
                    for key in ("visible", "showlegend"):
                        if key in args_list[0]:
                            local = args_list[0][key]
                            padded = (
                                [None] * trace_start
                                + local
                                + [None] * (total_traces - trace_start - len(local))
                            )
                            args_list[0][key] = padded
                extra_menus.append(um_dict)

        if fig.layout.annotations:
            for an in fig.layout.annotations:
                an_dict = an.to_plotly_json()
                if an_dict.get("xref") == "paper":
                    local_x = an_dict.get("x", 0)
                    an_dict["x"] = xdom[0] + local_x * dom_width
                extra_annots.append(an_dict)

    # Dynamic height/margin: taller when captions are below the plot.
    has_below_caption = any(
        _row_item(item)[2] and _row_item(item)[4] != "side"
        for item in row
    )
    height   = 560 if has_below_caption else 480
    b_margin = 160 if has_below_caption else 40

    combined.update_layout(
        font=dict(size=13),
        height=height,
        bargap=0.55,
        plot_bgcolor="white",
        paper_bgcolor="#f5f7fa",
        showlegend=True,
        legend=dict(font_size=12, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#ddd", borderwidth=1),
        margin=dict(t=80 if extra_menus else 60, b=b_margin, l=65, r=45),
        **({"updatemenus": extra_menus} if extra_menus else {}),
    )
    if extra_annots:
        for an in extra_annots:
            combined.add_annotation(**an)
    # Consistent axis styling across all subplots.
    combined.update_xaxes(gridcolor="#eaeaea",
                          title_font=dict(size=14, color="#444", family="Arial Black"))
    combined.update_yaxes(gridcolor="#eaeaea", zeroline=False,
                          title_font=dict(size=14, color="#444", family="Arial Black"))
    if shared_yaxes:
        combined.update_yaxes(showticklabels=True)
    return combined


def _write_html_report(path, row_figs, sample_name):
    """
    Combine per-row Plotly figures into a single self-contained HTML file.
    Plotly.js is loaded once via CDN; each figure renders with its own modebar.
    """
    import plotly.io as pio
    try:
        from importlib.metadata import version as _pkg_version
        __version__ = _pkg_version("tranquillyzer")
    except Exception:
        __version__ = "unknown"

    html_figs = []
    for i, fig in enumerate(row_figs):
        html_figs.append(pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs="cdn" if i == 0 else False,
            config={"displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        ))

    page_title = f"Tranquillyzer v{__version__} QC Report \u2014 {sample_name}"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "<!DOCTYPE html><html>"
            "<head><meta charset='utf-8'><style>"
            "body{background:#f5f7fa;font-family:Arial,sans-serif;margin:0;padding:16px}"
            "h1{text-align:center;font-size:18px;color:#333;margin:0 0 8px}"
            "</style></head>"
            f"<body><h1>{page_title}</h1>"
        )
        for fig_html in html_figs:
            fh.write(
                '<div style="background:white;border-radius:8px;'
                'box-shadow:0 1px 4px rgba(0,0,0,0.12);'
                'margin-bottom:16px;padding:8px">'
            )
            fh.write(fig_html)
            fh.write("</div>")
        fh.write("</body></html>")


# ─────────────────────────────────────────────────────────────────────────────
# metric / plot functions
# ─────────────────────────────────────────────────────────────────────────────


def _count_matching_readnames(path, pattern):
    """Count rows whose ReadName matches a regex pattern (lazy scan)."""
    if path is None:
        return 0
    schema = _probe_schema(path)
    if "ReadName" not in schema:
        return 0
    return (
        pl.scan_parquet(path)
        .select("ReadName")
        .filter(pl.col("ReadName").cast(pl.Utf8).str.contains(pattern))
        .select(pl.len())
        .collect()[0, 0]
    )


def _compute_summary(valid_path, invalid_path, vcols):
    """
    Compute total / valid / invalid / demuxed / ambiguous counts.

    When ``--split-concatenated`` was used, split fragments (``__frag*``)
    appear in the valid parquet and remainder entries (``__remainder``) in
    the invalid parquet.  These are separated so the summary reflects
    physical reads as the base unit.

    Uses at most 2 lazy scans (one per file) instead of 5 separate scans.
    """
    cell_col = _first_present(vcols, ["cell_id", "corrected_CBC"])
    has_cell = cell_col is not None

    # ── valid file: row count + fragment count + cell stats in one scan ─────
    n_valid_rows = n_fragments = n_demuxed = n_ambiguous = 0
    if valid_path is not None:
        exprs = [pl.len().alias("n_rows")]
        if "ReadName" in vcols:
            exprs.append(
                pl.col("ReadName").cast(pl.Utf8).str.contains(r"__frag\d+$").sum().alias("n_fragments")
            )
        if has_cell:
            exprs.append(_expr_is_demuxed(cell_col).sum().alias("n_demuxed"))
            exprs.append(_expr_is_ambiguous(cell_col).sum().alias("n_ambiguous"))
        row = pl.scan_parquet(valid_path).select(exprs).collect().row(0, named=True)
        n_valid_rows = row["n_rows"]
        n_fragments = row.get("n_fragments", 0)
        n_demuxed = row.get("n_demuxed", 0)
        n_ambiguous = row.get("n_ambiguous", 0)

    # ── invalid file: row count + remainder count in one scan ──────────────
    n_invalid_rows = n_concat_parents = 0
    if invalid_path is not None:
        icols = _probe_schema(invalid_path)
        i_exprs = [pl.len().alias("n_rows")]
        if "ReadName" in icols:
            i_exprs.append(
                pl.col("ReadName").cast(pl.Utf8).str.contains(r"__remainder$").sum().alias("n_remainder")
            )
        irow = pl.scan_parquet(invalid_path).select(i_exprs).collect().row(0, named=True)
        n_invalid_rows = irow["n_rows"]
        n_concat_parents = irow.get("n_remainder", 0)

    has_split        = n_fragments > 0
    n_valid_natural  = n_valid_rows - n_fragments
    n_invalid_true   = n_invalid_rows - n_concat_parents
    n_total_physical = n_valid_natural + n_concat_parents + n_invalid_true

    return {
        "n_total":           n_total_physical,
        "n_valid":           n_valid_natural,
        "n_invalid":         n_invalid_true,
        "n_concat_parents":  n_concat_parents,
        "n_fragments":       n_fragments,
        "n_effective_valid": n_valid_natural + n_fragments,
        "has_split":         has_split,
        "n_demuxed":         n_demuxed,
        "n_ambiguous":       n_ambiguous,
        "has_cell":          has_cell,
        "cell_col":          cell_col,
    }


def _plot_read_architecture(summary, sample_name):
    """
    Plot 1 — Read Architecture: Total / Valid / (Concatenated) / Invalid / (Effective Valid).

    When ``--split-concatenated`` produced fragments, two extra bars are shown:
    *Concatenated* (physical reads that were split) and *Effective Valid*
    (natural valid + recovered fragments).
    Returns (title, go.Figure, caption).
    """
    n_total    = summary["n_total"]
    n_valid    = summary["n_valid"]
    n_invalid  = summary["n_invalid"]
    has_split  = summary["has_split"]
    n_concat   = summary["n_concat_parents"]
    n_frags    = summary["n_fragments"]
    n_eff      = summary["n_effective_valid"]

    def _pct(n, d):
        return 100.0 * n / d if d > 0 else 0.0

    labels = ["Total", "Valid"]
    colors = ["#4C78A8", "#54A24B"]
    counts = [n_total, n_valid]
    text   = [
        f"{n_total:,}",
        f"{n_valid:,}<br>({_pct(n_valid, n_total):.1f}%)",
    ]

    if has_split:
        labels.append("Concatenated")
        colors.append("#EECA3B")
        counts.append(n_concat)
        text.append(
            f"{n_concat:,}<br>({_pct(n_concat, n_total):.1f}%)"
            f"<br>\u2192 {n_frags:,} frags"
        )

    labels.append("Invalid")
    colors.append("#E45756")
    counts.append(n_invalid)
    text.append(f"{n_invalid:,}<br>({_pct(n_invalid, n_total):.1f}%)")

    if has_split:
        labels.append("Effective Valid")
        colors.append("#2D8E2D")
        counts.append(n_eff)
        text.append(f"{n_eff:,}<br>({_pct(n_eff, n_total):.1f}%)")

    fig = go.Figure(go.Bar(
        x=labels, y=counts, text=text,
        textposition="outside", cliponaxis=False,
        textfont=dict(size=13),
        marker=dict(color=colors, line=dict(color="white", width=1)),
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickfont_size=13),
        yaxis=dict(tickfont_size=13, title="Read Count", gridcolor="#f0f0f0", zerolinecolor="#ddd"),
        plot_bgcolor="white", paper_bgcolor="white",
        uniformtext_minsize=13, uniformtext_mode="show",
    )

    caption = (
        "Reads classified by predicted segment order vs. expected library structure. "
        "Counts are based on physical reads (not annotation rows)."
        "\n\n"
        "<b>Valid</b>: single complete pattern.&nbsp;&nbsp;"
    )
    if has_split:
        caption += (
            "<b>Concatenated</b>: reads split into fragments "
            "(arrow shows recovered fragment count).&nbsp;&nbsp;"
        )
    caption += "<b>Invalid</b>: no identifiable pattern."
    if has_split:
        caption += (
            "&nbsp;&nbsp;<b>Effective Valid</b>: Valid + recovered fragments."
        )
    caption += "\n\n"

    return "Read Architecture", fig, caption, -0.14


def _plot_barcode_assignment(summary, sample_name):
    """
    Plot 2 — Barcode Assignment: Demuxed / Ambiguous bars.

    Denominator is ``n_effective_valid`` (natural valid + recovered fragments)
    when splitting was used, otherwise ``n_valid`` (natural valid only).
    Returns (title, go.Figure, caption), or None when no cell assignment data.
    """
    if not summary["has_cell"]:
        return None

    has_split  = summary["has_split"]
    n_denom    = summary["n_effective_valid"] if has_split else summary["n_valid"]
    n_demuxed  = summary["n_demuxed"]
    n_ambiguous = summary["n_ambiguous"]

    def _pct(n, d):
        return 100.0 * n / d if d > 0 else 0.0

    labels = ["Demuxed", "Ambiguous"]
    colors = ["#72B7B2", "#F58518"]
    counts = [n_demuxed, n_ambiguous]
    text   = [
        f"{n_demuxed:,}<br>({_pct(n_demuxed, n_denom):.1f}%)",
        f"{n_ambiguous:,}<br>({_pct(n_ambiguous, n_denom):.1f}%)",
    ]

    fig = go.Figure(go.Bar(
        x=labels, y=counts, text=text,
        textposition="outside", cliponaxis=False,
        textfont=dict(size=13),
        marker=dict(color=colors, line=dict(color="white", width=1)),
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickfont_size=13),
        yaxis=dict(title="Read Count", gridcolor="#f0f0f0", zerolinecolor="#ddd"),
        plot_bgcolor="white", paper_bgcolor="white",
        uniformtext_minsize=13, uniformtext_mode="show",
    )

    denom_label = "effective valid reads (valid + recovered fragments)" if has_split else "valid reads"
    caption = (
        f"From {denom_label}, each is assigned to a cell barcode via Levenshtein distance."
        "\n\n"
    )
    title = "Barcode Assignment (effective valid)" if has_split else "Barcode Assignment (valid reads)"
    return title, fig, caption, -0.14


def _plot_invalid_reasons(invalid_path):
    """
    Horizontal bar chart of the top-N reasons reads were classified as invalid.

    A Plotly dropdown lets the user switch between showing the top 5 / 10 / 15 /
    20 / 50 / All reasons.  Returns (title, go.Figure, caption) or None.
    """
    if invalid_path is None:
        return None

    icols = _probe_schema(invalid_path)
    if "reason" not in icols:
        return None

    load_cols = ["reason"] + (["ReadName"] if "ReadName" in icols else [])
    df = _scan_cols(invalid_path, load_cols, schema=icols).drop_nulls(subset=["reason"])
    if df.is_empty():
        return None

    # Exclude remainder entries from split-concatenated reads — they are
    # bookkeeping rows whose parent concatenated reason already appears
    # in the chart via the true invalid entries.
    if "ReadName" in df.columns:
        df = df.filter(~pl.col("ReadName").cast(pl.Utf8).str.contains(r"__remainder$"))
    if df.is_empty():
        return None

    vc = (
        df.group_by("reason")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    all_reasons = vc["reason"].to_list()
    all_counts  = vc["count"].to_list()
    total       = sum(all_counts)
    n_total     = len(all_reasons)

    _MAX_LABEL = 60   # truncate long reason strings for display

    def _unique_labels(reasons):
        """Truncate long reasons and disambiguate collisions with a counter."""
        raw = [r if len(r) <= _MAX_LABEL else r[:_MAX_LABEL - 1] + "…" for r in reasons]
        seen: dict[str, int] = {}
        out = []
        for lbl in raw:
            count = seen.get(lbl, 0)
            seen[lbl] = count + 1
            out.append(f"{lbl} ({count + 1})" if count else lbl)
        # Go back and fix the first occurrence if it turned out to have duplicates
        for i, lbl in enumerate(raw):
            if seen[lbl] > 1 and out[i] == lbl:
                out[i] = f"{lbl} (1)"
        return out

    labels  = _unique_labels(all_reasons)
    pcts    = [100 * c / total if total else 0 for c in all_counts]

    # Pre-slice for each preset; cap at actual count
    _PRESETS = [5, 10, 15, 20, 50]
    presets  = [n for n in _PRESETS if n < n_total] + [n_total]
    preset_labels = [f"Top {n}" if n < n_total else "All" for n in presets]

    def _slice(n):
        xs = labels[:n][::-1]       # reverse so largest is at top (horizontal bar)
        ys = all_counts[:n][::-1]
        ps = pcts[:n][::-1]
        hover = [
            f"<b>{all_reasons[i]}</b><br>Count: {all_counts[i]:,}  ({pcts[i]:.1f}%)"
            for i in range(n - 1, -1, -1)
        ]
        return xs, ys, ps, hover

    # Build initial trace (first preset)
    x0, y0, _, h0 = _slice(presets[0])
    fig = go.Figure(go.Bar(
        x=y0,
        y=x0,
        orientation="h",
        marker=dict(color="#E45756"),
        hovertext=h0,
        hoverinfo="text",
        showlegend=False,
    ))

    # Dropdown buttons — each updates x/y/hovertext and the y-axis range
    _MENU_STYLE = dict(
        type="dropdown", direction="down",
        bgcolor="white", bordercolor="#ccc", borderwidth=1,
        font=dict(size=12),
        showactive=True,
    )
    buttons = []
    for n, lbl in zip(presets, preset_labels):
        xs, ys, _, hs = _slice(n)
        buttons.append(dict(
            label=lbl,
            method="restyle",
            args=[{"x": [ys], "y": [xs], "hovertext": [hs]}],
        ))

    fig.update_layout(
        updatemenus=[dict(
            **_MENU_STYLE,
            x=0.0, xanchor="left",
            y=1.18, yanchor="top",
            buttons=buttons,
            active=0,
        )],
        annotations=[dict(
            text="Show:", x=0.0, xanchor="left",
            y=1.24, yanchor="top",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12, color="#444"),
        )],
        xaxis=dict(
            title="Read Count",
            gridcolor="#f0f0f0",
            tickfont_size=12,
        ),
        yaxis=dict(
            tickfont_size=11,
            automargin=True,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    caption = (
        f"Top reasons reads were classified as invalid ({n_total:,} distinct reasons, "
        f"{total:,} invalid reads total). Use the dropdown to adjust how many reasons are shown."
    )
    return "Invalid Read Reasons", fig, caption, -0.20


def _plot_edit_distances(valid_path, barcode_types, vcols):
    """
    One bar chart per barcode type showing the frequency of each minimum edit
    distance value.  Returns a list of ``(title, fig, caption)`` tuples.
    """
    _BAR_COLOR = "#4C78A8"

    results = []
    for bc in barcode_types:
        dist_col = f"corrected_{bc}_min_dist"
        if dist_col not in vcols:
            continue

        df = _scan_cols(valid_path, [dist_col], schema=vcols).drop_nulls()
        if df.is_empty():
            continue

        _MAX_DIST = 5
        vc = df[dist_col].cast(pl.Int64).value_counts().sort(dist_col)
        raw_distances = [int(d) for d in vc[dist_col].to_list()]
        raw_counts    = vc["count"].to_list()

        # Cap: group anything ≥ _MAX_DIST into a single "5+" bucket.
        capped: dict[int, int] = {}
        for d, c in zip(raw_distances, raw_counts):
            key = min(d, _MAX_DIST)
            capped[key] = capped.get(key, 0) + c
        distances = sorted(capped)
        counts    = [capped[d] for d in distances]
        total     = sum(counts)

        x_labels = [f"≥{_MAX_DIST}" if d == _MAX_DIST else str(d) for d in distances]
        text     = [f"{c:,}<br>({100*c/total:.1f}%)" if total else f"{c:,}" for c in counts]

        fig = go.Figure(go.Bar(
            x=x_labels,
            y=counts,
            text=text,
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=13),
            marker=dict(color=_BAR_COLOR, line=dict(color="white", width=1)),
            hovertemplate="Edit dist %{x}: %{y:,}<extra></extra>",
        ))
        fig.update_layout(
            xaxis=dict(title="Edit Distance", tickfont_size=13),
            yaxis=dict(title="Read Count", gridcolor="#f0f0f0", zerolinecolor="#ddd"),
            plot_bgcolor="white", paper_bgcolor="white",
        )

        caption = (
            f"Minimum Levenshtein distance between the observed <b>{bc}</b> sequence "
            "and the nearest whitelist barcode(s)."
            "\n\n"
            "<b>0</b>: exact match.&nbsp;&nbsp;"
            "<b>1–2</b>: corrected; within threshold.&nbsp;&nbsp;"
            "<b>3+</b>: uncorrected; above threshold"
        )
        results.append((f"Edit Distance — {bc}", fig, caption, -0.18))

    return results


def _build_multires_histograms(group_dfs, max_len, all_nbins):
    """Compute histograms at multiple resolutions efficiently.

    Bins the data once at the finest granularity (largest n_bins), then
    derives all coarser histograms by summing adjacent fine bins — avoiding
    repeated ``group_by`` operations on the full dataset.

    Returns ``{nb_idx: {group_name: {bin_edge: count}}}``.
    """
    finest_bw = max(1, max_len // all_nbins[-1])

    def _bin_counts_fine(df):
        return dict(
            df.with_columns(
                (pl.col("read_length") // finest_bw * finest_bw).alias("bin")
            )
            .group_by("bin").len().sort("bin").rows()
        )

    fine_counts = {nm: _bin_counts_fine(group_dfs[nm]) for nm in group_dfs}

    result = {}
    for nb_idx, n_bins in enumerate(all_nbins):
        bw = max(1, max_len // n_bins)
        result[nb_idx] = {}
        for nm, fc in fine_counts.items():
            coarse: dict[int, int] = {}
            for fine_bin, cnt in fc.items():
                cb = (fine_bin // bw) * bw
                coarse[cb] = coarse.get(cb, 0) + cnt
            result[nb_idx][nm] = coarse
    return result


def _plot_read_length_dist(valid_path, invalid_path, vcols, bin_width):
    """
    Line chart of read-length distribution for five read subsets.
    Provides interactive dropdowns to change bin width and cap the x-axis.
    Returns (title, go.Figure, caption) or None.
    """
    if "read_length" not in vcols:
        return None

    _COLORS = {
        "Total":              "#888888",
        "Invalid":            "#E45756",
        "Valid":              "#54A24B",
        "Valid — from split": "#8BC34A",
        "Valid — Demuxed":    "#72B7B2",
        "Valid — Ambiguous":  "#F58518",
    }
    _NBIN_PRESETS = [50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000,
                     200_000, 300_000, 400_000, 500_000, 1_000_000]

    cell_col  = _first_present(list(vcols), ["cell_id", "corrected_CBC"])
    load_cols = ["read_length", "ReadName"] + ([cell_col] if cell_col else [])
    valid_df  = _scan_cols(valid_path,   load_cols, schema=vcols) if valid_path   else pl.DataFrame()
    inv_df    = _scan_cols(invalid_path, ["read_length"])        if invalid_path else pl.DataFrame()

    # Ensure read_length is numeric (chunk parquets may store it as str)
    if not valid_df.is_empty() and "read_length" in valid_df.columns:
        valid_df = valid_df.with_columns(pl.col("read_length").cast(pl.Int64, strict=False))
    if not inv_df.is_empty() and "read_length" in inv_df.columns:
        inv_df = inv_df.with_columns(pl.col("read_length").cast(pl.Int64, strict=False))

    # Max read length (needed to compute bin widths)
    max_read_len = 0
    if not valid_df.is_empty() and "read_length" in valid_df.columns:
        max_read_len = max(max_read_len, int(valid_df["read_length"].max()))
    if not inv_df.is_empty() and "read_length" in inv_df.columns:
        max_read_len = max(max_read_len, int(inv_df["read_length"].max()))
    if max_read_len == 0:
        return None

    # Build per-group source dataframes (filtered once, reused for every n_bins)
    group_dfs: dict[str, pl.DataFrame] = {}
    parts = []
    if not valid_df.is_empty():
        parts.append(valid_df.select("read_length"))
    if not inv_df.is_empty() and "read_length" in inv_df.columns:
        parts.append(inv_df.select("read_length"))
    if parts:
        group_dfs["Total"] = pl.concat(parts)
    if not inv_df.is_empty() and "read_length" in inv_df.columns:
        group_dfs["Invalid"] = inv_df.select("read_length")
    if not valid_df.is_empty():
        group_dfs["Valid"] = valid_df.select("read_length")
    # "Valid — from split": fragment entries recovered from concatenated reads
    if not valid_df.is_empty() and "ReadName" in valid_df.columns:
        split_df = valid_df.filter(pl.col("ReadName").cast(pl.Utf8).str.contains(r"__frag\d+$"))
        if not split_df.is_empty():
            group_dfs["Valid — from split"] = split_df.select("read_length")

    if cell_col and not valid_df.is_empty() and cell_col in valid_df.columns:
        cell_utf8 = pl.col(cell_col).cast(pl.Utf8)
        vd_df = valid_df.filter(
            pl.col(cell_col).is_not_null() & (cell_utf8 != "") & (cell_utf8 != "ambiguous")
        )
        if not vd_df.is_empty():
            group_dfs["Valid — Demuxed"] = vd_df.select("read_length")
        va_df = valid_df.filter(cell_utf8 == "ambiguous")
        if not va_df.is_empty():
            group_dfs["Valid — Ambiguous"] = va_df.select("read_length")

    if not group_dfs:
        return None

    group_names = list(group_dfs.keys())
    n_groups    = len(group_names)

    # Total-bins options: only keep presets that give ≥1 bp/bin
    all_nbins = [nb for nb in _NBIN_PRESETS if max_read_len // nb >= 1] or [100]
    n_bw      = len(all_nbins)
    # Default: n_bins closest to max_read_len / bin_width
    target    = max_read_len / bin_width if bin_width > 0 else 200
    default_bw_idx = min(range(n_bw), key=lambda i: abs(all_nbins[i] - target))

    # ── add all traces (n_groups × n_bins options); only default visible ──────
    multires = _build_multires_histograms(group_dfs, max_read_len, all_nbins)
    fig = go.Figure()
    for nb_idx, n_bins in enumerate(all_nbins):
        bw         = max(1, max_read_len // n_bins)
        all_counts = multires[nb_idx]
        all_bins   = sorted({b for c in all_counts.values() for b in c})
        visible    = (nb_idx == default_bw_idx)
        for name in group_names:
            counts = all_counts[name]
            fig.add_trace(go.Scatter(
                x=[int(b) + bw / 2 for b in all_bins],
                y=[counts.get(b, 0) for b in all_bins],
                mode="lines",
                name=name,
                visible=visible,
                showlegend=visible,
                line=dict(color=_COLORS.get(name, "#333"), width=2),
                hovertemplate=(
                    f"<b>{name}</b><br>Length: %{{x:,.0f}} bp"
                    f"<br>Count: %{{y:,}}<extra></extra>"
                ),
            ))

    # ── total-bins dropdown (restyle: toggle visible + showlegend) ────────────
    bw_buttons = []
    for nb_idx, n_bins in enumerate(all_nbins):
        vis  = [False] * (n_groups * n_bw)
        sleg = [False] * (n_groups * n_bw)
        for j in range(n_groups):
            vis [nb_idx * n_groups + j] = True
            sleg[nb_idx * n_groups + j] = True
        bw_buttons.append(dict(
            method="restyle",
            label=f"{n_bins:,}",
            args=[{"visible": vis, "showlegend": sleg}],
        ))

    # ── x-axis cap dropdown ───────────────────────────────────────────────────
    _CAP_PRESETS = [500, 1_000, 2_000, 3_000, 5_000, 7_500, 10_000, 15_000,
                    20_000, 30_000, 50_000, 75_000, 100_000, 200_000, 500_000]
    caps = [p for p in _CAP_PRESETS if p < max_read_len]
    cap_buttons = [
        dict(
            method="relayout",
            label=f"≤ {cap:,} bp",
            args=[{"xaxis.range": [0, cap], "xaxis.autorange": False}],
        )
        for cap in caps
    ]
    cap_buttons.append(dict(
        method="relayout",
        label="All",
        args=[{"xaxis.autorange": True}],
    ))

    _MENU = dict(type="dropdown", direction="down", font=dict(size=12),
                 showactive=True, bgcolor="white", bordercolor="#ccc",
                 y=1.18, yanchor="top")

    fig.update_layout(
        xaxis=dict(title="Read Length (bp)", tickfont_size=13, gridcolor="#f0f0f0"),
        yaxis=dict(title="Read Count",       tickfont_size=13, gridcolor="#f0f0f0",
                   zerolinecolor="#ddd"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(font_size=12, bgcolor="rgba(255,255,255,0.8)"),
        updatemenus=[
            dict(**_MENU, active=default_bw_idx,
                 x=0.0,  xanchor="left", buttons=bw_buttons),
            dict(**_MENU, active=len(cap_buttons) - 1,
                 x=0.12, xanchor="left", buttons=cap_buttons),
        ],
        annotations=[
            dict(text="Bins:", x=0.0,  xanchor="left", y=1.24, yanchor="top",
                 xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12, color="#444")),
            dict(text="Cap x-axis:", x=0.12, xanchor="left", y=1.24, yanchor="top",
                 xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12, color="#444")),
        ],
    )

    caption = (
        "Read-length distribution. Total bins and x-axis cap are adjustable via the dropdowns above."
        "\n\n"
        "<b>Total</b>: all reads (valid + invalid).&nbsp;&nbsp;"
        "<b>Invalid</b>: incorrect segment architecture.&nbsp;&nbsp;"
        "<b>Valid</b>: correct segment architecture.&nbsp;&nbsp;"
        "<b>Valid — from split</b>: fragments recovered from concatenated reads "
        "(subset of Valid; only shown when splitting was used).&nbsp;&nbsp;"
        "<b>Valid — Demuxed</b>: unambiguously assigned to a cell.&nbsp;&nbsp;"
        "<b>Valid — Ambiguous</b>: ambiguous cell assignment."
    )
    return "Read-Length Distribution", fig, caption, -0.19


def _plot_cdna_length_dist(valid_path, vcols, bin_width):
    """
    Line chart of cDNA-length distribution for valid reads only.
    Same interactive dropdowns as the read-length plot (bin width, x-cap).
    Returns (title, go.Figure, caption) or None.
    """
    if "cDNA_Starts" not in vcols or "cDNA_Ends" not in vcols:
        return None

    _COLORS = {
        "Valid":              "#54A24B",
        "Valid — from split": "#8BC34A",
        "Valid — Demuxed":    "#72B7B2",
        "Valid — Ambiguous":  "#F58518",
    }
    _NBIN_PRESETS = [50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000,
                     200_000, 300_000, 400_000, 500_000, 1_000_000]

    cell_col  = _first_present(list(vcols), ["cell_id", "corrected_CBC"])
    load_cols = ["cDNA_Starts", "cDNA_Ends", "ReadName"] + ([cell_col] if cell_col else [])
    valid_df  = _scan_cols(valid_path, load_cols, schema=vcols) if valid_path else pl.DataFrame()

    if valid_df.is_empty() or "cDNA_Starts" not in valid_df.columns or "cDNA_Ends" not in valid_df.columns:
        return None

    valid_df = _add_cdna_length(valid_df)
    valid_df = valid_df.filter(pl.col("cDNA_length").is_not_null() & (pl.col("cDNA_length") > 0))

    if valid_df.is_empty():
        return None

    max_len = int(valid_df["cDNA_length"].max())
    if max_len == 0:
        return None

    # Build per-group source dataframes
    group_dfs: dict[str, pl.DataFrame] = {}
    group_dfs["Valid"] = valid_df.select("cDNA_length").rename({"cDNA_length": "read_length"})

    if "ReadName" in valid_df.columns:
        split_df = valid_df.filter(pl.col("ReadName").cast(pl.Utf8).str.contains(r"__frag\d+$"))
        if not split_df.is_empty():
            group_dfs["Valid — from split"] = split_df.select("cDNA_length").rename({"cDNA_length": "read_length"})

    if cell_col and cell_col in valid_df.columns:
        cell_utf8 = pl.col(cell_col).cast(pl.Utf8)
        vd_df = valid_df.filter(
            pl.col(cell_col).is_not_null() & (cell_utf8 != "") & (cell_utf8 != "ambiguous")
        )
        if not vd_df.is_empty():
            group_dfs["Valid — Demuxed"] = vd_df.select("cDNA_length").rename({"cDNA_length": "read_length"})
        va_df = valid_df.filter(cell_utf8 == "ambiguous")
        if not va_df.is_empty():
            group_dfs["Valid — Ambiguous"] = va_df.select("cDNA_length").rename({"cDNA_length": "read_length"})

    if not group_dfs:
        return None

    group_names = list(group_dfs.keys())
    n_groups    = len(group_names)

    all_nbins = [nb for nb in _NBIN_PRESETS if max_len // nb >= 1] or [100]
    n_bw      = len(all_nbins)
    target    = max_len / bin_width if bin_width > 0 else 200
    default_bw_idx = min(range(n_bw), key=lambda i: abs(all_nbins[i] - target))

    multires = _build_multires_histograms(group_dfs, max_len, all_nbins)
    fig = go.Figure()
    for nb_idx, n_bins in enumerate(all_nbins):
        bw         = max(1, max_len // n_bins)
        all_counts = multires[nb_idx]
        all_bins   = sorted({b for c in all_counts.values() for b in c})
        visible    = (nb_idx == default_bw_idx)
        for name in group_names:
            counts = all_counts[name]
            fig.add_trace(go.Scatter(
                x=[int(b) + bw / 2 for b in all_bins],
                y=[counts.get(b, 0) for b in all_bins],
                mode="lines",
                name=name,
                visible=visible,
                showlegend=visible,
                line=dict(color=_COLORS.get(name, "#333"), width=2),
                hovertemplate=(
                    f"<b>{name}</b><br>Length: %{{x:,.0f}} bp"
                    f"<br>Count: %{{y:,}}<extra></extra>"
                ),
            ))

    bw_buttons = []
    for nb_idx, n_bins in enumerate(all_nbins):
        vis  = [False] * (n_groups * n_bw)
        sleg = [False] * (n_groups * n_bw)
        for j in range(n_groups):
            vis [nb_idx * n_groups + j] = True
            sleg[nb_idx * n_groups + j] = True
        bw_buttons.append(dict(
            method="restyle",
            label=f"{n_bins:,}",
            args=[{"visible": vis, "showlegend": sleg}],
        ))

    _CAP_PRESETS = [500, 1_000, 2_000, 3_000, 5_000, 7_500, 10_000, 15_000,
                    20_000, 30_000, 50_000, 75_000, 100_000, 200_000, 500_000]
    caps = [p for p in _CAP_PRESETS if p < max_len]
    cap_buttons = [
        dict(
            method="relayout",
            label=f"≤ {cap:,} bp",
            args=[{"xaxis.range": [0, cap], "xaxis.autorange": False}],
        )
        for cap in caps
    ]
    cap_buttons.append(dict(
        method="relayout",
        label="All",
        args=[{"xaxis.autorange": True}],
    ))

    _MENU = dict(type="dropdown", direction="down", font=dict(size=12),
                 showactive=True, bgcolor="white", bordercolor="#ccc",
                 y=1.18, yanchor="top")

    fig.update_layout(
        xaxis=dict(title="cDNA Length (bp)", tickfont_size=13, gridcolor="#f0f0f0"),
        yaxis=dict(title="Read Count",       tickfont_size=13, gridcolor="#f0f0f0",
                   zerolinecolor="#ddd"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(font_size=12, bgcolor="rgba(255,255,255,0.8)"),
        updatemenus=[
            dict(**_MENU, active=default_bw_idx,
                 x=0.0,  xanchor="left", buttons=bw_buttons),
            dict(**_MENU, active=len(cap_buttons) - 1,
                 x=0.12, xanchor="left", buttons=cap_buttons),
        ],
        annotations=[
            dict(text="Bins:", x=0.0,  xanchor="left", y=1.24, yanchor="top",
                 xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12, color="#444")),
            dict(text="Cap x-axis:", x=0.12, xanchor="left", y=1.24, yanchor="top",
                 xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12, color="#444")),
        ],
    )

    caption = (
        "cDNA-length distribution for valid reads. Total bins and x-axis cap are adjustable via the dropdowns above."
        "\n\n"
        "<b>Valid</b>: correct segment architecture.&nbsp;&nbsp;"
        "<b>Valid — from split</b>: fragments recovered from concatenated reads "
        "(subset of Valid; only shown when splitting was used).&nbsp;&nbsp;"
        "<b>Valid — Demuxed</b>: unambiguously assigned to a cell.&nbsp;&nbsp;"
        "<b>Valid — Ambiguous</b>: ambiguous cell assignment."
    )
    return "cDNA-Length Distribution", fig, caption, -0.19


# ─────────────────────────────────────────────────────────────────────────────
# barcode knee plot  (transcripts vs barcodes, both log-scale)
# ─────────────────────────────────────────────────────────────────────────────


def _plot_knee(valid_path, vcols):
    """
    Knee / elbow plot: rank barcodes by read count (descending).

    X-axis (log) : barcode rank
    Y-axis (log) : read count per barcode (proxy for transcripts; UMI
                   deduplication is performed at the BAM level and is not
                   reflected in the parquet annotation files)

    Returns (title, go.Figure, caption) or None.
    """
    bc_col = _first_present(list(vcols), ["cell_id", "corrected_CBC"])
    if bc_col is None:
        return None

    df = _scan_cols(valid_path, [bc_col], schema=vcols)
    if df.is_empty():
        return None

    # Only demuxed cell IDs — exclude "ambiguous" / null entries
    df = df.filter(_expr_is_demuxed(bc_col))
    if df.is_empty():
        return None

    counts = (
        df.group_by(bc_col)
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    ranks = list(range(1, len(counts) + 1))
    ns    = counts["n"].to_list()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ranks,
        y=ns,
        mode="lines",
        line=dict(color="#54A24B", width=2),
        name="Reads per barcode",
        hovertemplate="Rank %{x:,}<br>Reads: %{y:,}<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(
            title="Barcodes (rank)",
            type="log", dtick=1,
            tickfont_size=13,
            gridcolor="#f0f0f0",
            minor=dict(showgrid=False, ticks="", nticks=0),
        ),
        yaxis=dict(
            title="Reads (transcripts proxy)",
            type="log", dtick=1,
            tickfont_size=13,
            gridcolor="#f0f0f0",
            zerolinecolor="#ddd",
            minor=dict(showgrid=False, ticks="", nticks=0),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )

    caption = (
        f"Read count per cell after barcode correction against the discovered whitelist. "
        f"{len(ranks):,} cells with unambiguous barcode assignments ranked by read count (descending); "
        "ambiguous and unmatched reads excluded. Both axes are log-scaled. "
        "Note: UMI deduplication is applied at the BAM level and is not reflected here."
    )
    return "Reads per Cell", fig, caption, -0.14


def _plot_knee_whitelist_free(metadata_dir):
    """Barcode rank plot from whitelist-free discovery artifacts.

    Uses barcode_counts.tsv (produced by generate-whitelist) to plot all
    observed barcodes ranked by read count, with a vertical knee threshold line.
    Returns (title, go.Figure, caption, caption_y) or None.
    """
    counts_path = os.path.join(metadata_dir, "barcode_counts.tsv")
    if not os.path.exists(counts_path):
        return None

    df = pl.read_csv(counts_path, separator="\t")
    if df.is_empty() or "read_count" not in df.columns:
        return None

    sorted_counts = df.sort("read_count", descending=True)["read_count"].to_list()
    ranks = list(range(1, len(sorted_counts) + 1))

    # Knee position: count of canonical + merged barcodes (i.e. not "below_knee")
    n_knee = df.filter(~pl.col("status").str.starts_with("below_knee")).height

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ranks, y=sorted_counts, mode="lines",
        line=dict(color="#54A24B", width=2),
        name="Reads per barcode",
        hovertemplate="Rank %{x:,}<br>Reads: %{y:,}<extra></extra>",
    ))
    # Vertical knee threshold line (Scatter trace — add_vline doesn't render on log axes)
    y_min, y_max = sorted_counts[-1], sorted_counts[0]
    fig.add_trace(go.Scatter(
        x=[n_knee, n_knee], y=[y_min, y_max], mode="lines",
        line=dict(color="#E45756", width=1.5, dash="dash"),
        name=f"Knee ({n_knee:,})", showlegend=True,
        hovertemplate=f"Knee: {n_knee:,} barcodes<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(title="Barcode rank", type="log", dtick=1, tickfont_size=13, gridcolor="#f0f0f0",
                   minor=dict(showgrid=False, ticks="", nticks=0)),
        yaxis=dict(title="Read count", type="log", dtick=1, tickfont_size=13, gridcolor="#f0f0f0", zerolinecolor="#ddd",
                   minor=dict(showgrid=False, ticks="", nticks=0)),
        plot_bgcolor="white", paper_bgcolor="white",
    )

    caption = (
        f"All {len(ranks):,} observed canonicalized barcodes from valid reads, ranked by read count (descending). "
        f"Knee-point detection on the log-log rank curve identifies {n_knee:,} high-confidence cell barcodes "
        "(vertical line). Near-duplicate barcodes within 1 edit distance are merged into canonical entries. "
        "Barcodes to the right of the threshold are considered sequencing noise or artifacts. Both axes are log-scaled."
    )
    return "Barcode Discovery", fig, caption, -0.14


# ─────────────────────────────────────────────────────────────────────────────
# per-cell read-length summary (median line + IQR band)
# ─────────────────────────────────────────────────────────────────────────────


def _plot_read_length_per_cell(valid_path, vcols):
    """
    Line plot of read-length statistics per cell-id.

    X-axis : cell_id, sorted numerically (with "ambiguous" appended last).
    Y-axis : read length.
    Traces :
      - Shaded band  Q1 → Q3  (fill between)
      - Line         median

    Returns (title, go.Figure, caption) or None.
    """
    cell_col = _first_present(list(vcols), ["cell_id", "corrected_CBC"])
    if cell_col is None or "read_length" not in vcols:
        return None

    # Push the group-by aggregation into the lazy scan — polars never fully
    # materialises the per-read table, keeping peak memory and time low.
    stats = (
        pl.scan_parquet(valid_path)
        .select([cell_col, "read_length"])
        .with_columns(
            pl.col(cell_col).cast(pl.Utf8).alias("_cid"),
            pl.col("read_length").cast(pl.Int64, strict=False),
        )
        .filter(pl.col("_cid") != "")
        .group_by("_cid")
        .agg([
            pl.col("read_length").quantile(0.25, interpolation="midpoint").alias("q1"),
            pl.col("read_length").median().alias("median"),
            pl.col("read_length").quantile(0.75, interpolation="midpoint").alias("q3"),
            pl.col("read_length").count().alias("n"),
        ])
        .collect()
    )

    if stats.is_empty():
        return None

    # Sort by median read length descending; "ambiguous" appended last — done
    # inside polars to avoid slow Python list sort on large cell counts.
    stats = (
        stats
        .with_columns((pl.col("_cid") == "ambiguous").alias("_is_ambig"))
        .sort(["_is_ambig", "median"], descending=[False, True])
        .drop("_is_ambig")
    )
    rows = stats.to_dicts()

    x        = [r["_cid"]  for r in rows]
    medians  = [r["median"] for r in rows]
    q1s      = [r["q1"]     for r in rows]
    q3s      = [r["q3"]     for r in rows]

    _BAND_COLOR   = "rgba(84,162,75,0.25)"   # green, semi-transparent
    _MEDIAN_COLOR = "#54A24B"

    # IQR band: upper boundary then reversed lower boundary (Plotly fill trick)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=q3s + q1s[::-1],
        fill="toself",
        fillcolor=_BAND_COLOR,
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="IQR (Q1–Q3)",
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=medians,
        mode="lines+markers",
        line=dict(color=_MEDIAN_COLOR, width=2),
        marker=dict(size=4, color=_MEDIAN_COLOR),
        name="Median",
        hovertemplate=(
            "<b>Cell %{x}</b><br>"
            "Median: %{y:,.0f} bp<extra></extra>"
        ),
    ))

    n_cells = len([r for r in rows if r["_cid"] != "ambiguous"])
    fig.update_layout(
        xaxis=dict(
            title=f"Cell ID — sorted by median read length  ({n_cells:,} demuxed + ambiguous)",
            tickfont_size=11,
            showticklabels=len(x) <= 200,   # hide labels when too crowded
            gridcolor="#f0f0f0",
        ),
        yaxis=dict(
            title="Read Length (bp)",
            tickfont_size=13,
            gridcolor="#f0f0f0",
            zerolinecolor="#ddd",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(font_size=12, bgcolor="rgba(255,255,255,0.8)"),
    )

    caption = (
        "Per-cell read-length summary for demuxed valid reads (ambiguous appended last). "
        "The line shows the median read length; the shaded band spans the interquartile range (Q1–Q3). "
        "Cell IDs are sorted by median read length (descending); ambiguous reads are appended last."
    )
    return "Read Length per Cell", fig, caption, -0.14


def _plot_cdna_length_per_cell(valid_path, vcols):
    """
    Line plot of cDNA-length statistics per cell-id.

    Same structure as ``_plot_read_length_per_cell`` but uses cDNA length
    (``cDNA_Ends - cDNA_Starts``) instead of read length, with a green colour
    scheme for visual distinction.

    Returns (title, go.Figure, caption) or None.
    """
    cell_col = _first_present(list(vcols), ["cell_id", "corrected_CBC"])
    if cell_col is None or "cDNA_Starts" not in vcols or "cDNA_Ends" not in vcols:
        return None

    stats = (
        pl.scan_parquet(valid_path)
        .select([cell_col, "cDNA_Starts", "cDNA_Ends"])
        .with_columns(
            pl.col(cell_col).cast(pl.Utf8).alias("_cid"),
            (pl.col("cDNA_Ends").cast(pl.Int64) - pl.col("cDNA_Starts").cast(pl.Int64)).alias("cDNA_length"),
        )
        .filter((pl.col("_cid") != "") & pl.col("cDNA_length").is_not_null() & (pl.col("cDNA_length") > 0))
        .group_by("_cid")
        .agg([
            pl.col("cDNA_length").quantile(0.25, interpolation="midpoint").alias("q1"),
            pl.col("cDNA_length").median().alias("median"),
            pl.col("cDNA_length").quantile(0.75, interpolation="midpoint").alias("q3"),
            pl.col("cDNA_length").count().alias("n"),
        ])
        .collect()
    )

    if stats.is_empty():
        return None

    stats = (
        stats
        .with_columns((pl.col("_cid") == "ambiguous").alias("_is_ambig"))
        .sort(["_is_ambig", "median"], descending=[False, True])
        .drop("_is_ambig")
    )
    rows = stats.to_dicts()

    x        = [r["_cid"]   for r in rows]
    medians  = [r["median"]  for r in rows]
    q1s      = [r["q1"]      for r in rows]
    q3s      = [r["q3"]      for r in rows]

    _BAND_COLOR   = "rgba(84,162,75,0.25)"   # green, semi-transparent
    _MEDIAN_COLOR = "#54A24B"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=q3s + q1s[::-1],
        fill="toself",
        fillcolor=_BAND_COLOR,
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="IQR (Q1–Q3)",
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=medians,
        mode="lines+markers",
        line=dict(color=_MEDIAN_COLOR, width=2),
        marker=dict(size=4, color=_MEDIAN_COLOR),
        name="Median",
        hovertemplate=(
            "<b>Cell %{x}</b><br>"
            "Median: %{y:,.0f} bp<extra></extra>"
        ),
    ))

    n_cells = len([r for r in rows if r["_cid"] != "ambiguous"])
    fig.update_layout(
        xaxis=dict(
            title=f"Cell ID — sorted by median cDNA length  ({n_cells:,} demuxed + ambiguous)",
            tickfont_size=11,
            showticklabels=len(x) <= 200,
            gridcolor="#f0f0f0",
        ),
        yaxis=dict(
            title="cDNA Length (bp)",
            tickfont_size=13,
            gridcolor="#f0f0f0",
            zerolinecolor="#ddd",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(font_size=12, bgcolor="rgba(255,255,255,0.8)"),
    )

    caption = (
        "Per-cell cDNA-length summary for demuxed valid reads (ambiguous appended last). "
        "The line shows the median cDNA length; the shaded band spans the interquartile range (Q1–Q3). "
        "Cell IDs are sorted by median cDNA length (descending); ambiguous reads are appended last."
    )
    return "cDNA Length per Cell", fig, caption, -0.14


# ─────────────────────────────────────────────────────────────────────────────
# Segment length box plots
# ─────────────────────────────────────────────────────────────────────────────


def _detect_segments(vcols):
    """
    Return segment names that have both ``{name}_Starts`` and ``{name}_Ends``
    columns in *vcols*.  Order follows the canonical pipeline layout.
    """
    _ORDERED = ["5p", "CBC", "UMI", "SLS", "polyT", "cDNA", "polyA", "3p", "random_s", "random_e"]
    found = []
    extra = set()
    for col in vcols:
        if col.endswith("_Starts"):
            seg = col[:-len("_Starts")]
            if f"{seg}_Ends" in vcols:
                if seg in _ORDERED:
                    found.append(seg)
                else:
                    extra.add(seg)
    # Return in canonical order, then any extras alphabetically.
    ordered = [s for s in _ORDERED if s in found]
    ordered.extend(sorted(extra))
    return ordered


def _load_segment_lengths(path, segments, vcols):
    """
    Load segment start/end columns once and compute box-plot statistics.

    Returns (all_stats, demuxed_stats) where each is a
    ``dict[str, dict]`` mapping segment name → ``{n, min, q1, median, q3, max}``.
    ``demuxed_stats`` is ``None`` when no cell column is available.
    """
    starts_cols = [f"{s}_Starts" for s in segments]
    ends_cols = [f"{s}_Ends" for s in segments]
    load_cols = starts_cols + ends_cols

    # Also load _Sequences columns for barcode segments (CBC, CBC1, CBC2, …)
    seq_segments = {s for s in segments if f"{s}_Sequences" in vcols}
    for s in seq_segments:
        load_cols.append(f"{s}_Sequences")

    cell_col = _first_present(vcols, ["cell_id", "corrected_CBC"])
    if cell_col:
        load_cols.append(cell_col)

    df = _scan_cols(path, load_cols, schema=vcols)
    if df.is_empty():
        return {}, None

    def _compute(sub_df):
        result = {}
        for seg in segments:
            seq_col = f"{seg}_Sequences"
            # For segments with a _Sequences column, use string length
            if seg in seq_segments and seq_col in sub_df.columns:
                try:
                    len_col = (
                        sub_df.select(
                            pl.col(seq_col).cast(pl.Utf8).str.len_chars().alias("_len")
                        )
                        .drop_nulls()
                        .filter(pl.col("_len") > 0)
                    )
                except Exception:
                    continue
            else:
                sc, ec = f"{seg}_Starts", f"{seg}_Ends"
                if sc not in sub_df.columns or ec not in sub_df.columns:
                    continue
                try:
                    len_col = (
                        sub_df.select(
                            (pl.col(ec).cast(pl.Int64) - pl.col(sc).cast(pl.Int64)).alias("_len")
                        )
                        .drop_nulls()
                        .filter(pl.col("_len") >= 0)
                    )
                except Exception:
                    try:
                        len_col = (
                            sub_df.select(pl.col(sc).cast(pl.Utf8).alias("_s"),
                                          pl.col(ec).cast(pl.Utf8).alias("_e"))
                            .with_columns(
                                pl.col("_s").str.split(", ").list.eval(pl.element().cast(pl.Int64)).alias("_si"),
                                pl.col("_e").str.split(", ").list.eval(pl.element().cast(pl.Int64)).alias("_ei"),
                            )
                            .select(
                                (pl.col("_ei") - pl.col("_si")).list.sum().alias("_len")
                            )
                            .drop_nulls()
                            .filter(pl.col("_len") >= 0)
                        )
                    except Exception:
                        continue
            if len_col.is_empty():
                continue
            s = len_col["_len"]
            q1 = s.quantile(0.25, interpolation="midpoint")
            q3 = s.quantile(0.75, interpolation="midpoint")
            result[seg] = dict(
                n=len(s), min=s.min(), q1=q1,
                median=s.median(), q3=q3, max=s.max(),
            )
        return result

    all_stats = _compute(df)

    demuxed_stats = None
    if cell_col and cell_col in df.columns:
        demux_df = df.filter(_expr_is_demuxed(cell_col))
        if not demux_df.is_empty():
            demuxed_stats = _compute(demux_df)

    return all_stats, demuxed_stats


def _make_segment_boxplot(seg_stats, color, title, label):
    """Build a horizontal box plot figure from precomputed segment stats."""
    if not seg_stats:
        return None

    fig = go.Figure()
    n_reads = 0
    for seg in reversed(list(seg_stats.keys())):
        st = seg_stats[seg]
        n_reads = max(n_reads, st["n"])
        iqr = st["q3"] - st["q1"]
        lower = max(st["min"], st["q1"] - 1.5 * iqr)
        upper = min(st["max"], st["q3"] + 1.5 * iqr)
        fig.add_trace(go.Box(
            y=[seg],
            q1=[st["q1"]],
            median=[st["median"]],
            q3=[st["q3"]],
            lowerfence=[lower],
            upperfence=[upper],
            orientation="h",
            name=seg,
            marker=dict(color=color, opacity=0.85),
            line=dict(color=color),
            fillcolor=color,
            opacity=0.85,
            showlegend=False,
            hoverinfo="skip",
        ))
        # Hover marker at median — only fires when pointer is on the box
        fig.add_trace(go.Scatter(
            x=[st["median"]],
            y=[seg],
            mode="markers",
            marker=dict(size=12, opacity=0),
            showlegend=False,
            hovertemplate=(
                f"<b>{seg}</b><br>"
                f"Median: {st['median']:,.0f} bp<br>"
                f"Q1: {st['q1']:,.0f} bp<br>"
                f"Q3: {st['q3']:,.0f} bp<br>"
                f"Lower fence: {lower:,.0f} bp<br>"
                f"Upper fence: {upper:,.0f} bp<br>"
                f"n = {st['n']:,}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis=dict(title="Length (bp)", tickfont_size=13, gridcolor="#f0f0f0"),
        yaxis=dict(tickfont_size=13, gridcolor="#f0f0f0"),
        plot_bgcolor="white", paper_bgcolor="white",
    )

    caption = (
        f"Per-segment length distributions across {n_reads:,} {label} reads. "
        "Each box shows the median, IQR, and whiskers (1.5x IQR)."
    )
    return title, fig, caption, -0.14


def _plot_segment_lengths(valid_path, vcols):
    """
    Compute segment lengths once and return two plot tuples:
    one for all valid reads and one for demuxed reads.

    Returns (all_valid_plot, demuxed_plot) — each is
    ``(title, fig, caption, caption_y)`` or ``None``.
    """
    segments = _detect_segments(vcols)
    if not segments:
        return None, None

    all_lengths, demuxed_lengths = _load_segment_lengths(valid_path, segments, vcols)

    all_plot = _make_segment_boxplot(
        all_lengths, "#2D8E2D", "Segment Lengths (All Valid Reads)", "valid")
    demux_plot = _make_segment_boxplot(
        demuxed_lengths, "#72B7B2", "Segment Lengths (Demuxed Reads)", "demuxed"
    ) if demuxed_lengths else None

    return all_plot, demux_plot


# ─────────────────────────────────────────────────────────────────────────────
# BAM-dependent metrics
# ─────────────────────────────────────────────────────────────────────────────


def _collect_bam_per_cell_stats(bam_path, threads=4):
    """
    Collect per-cell alignment statistics from a dup-marked BAM.

    Uses **samtools** (C, multi-threaded BAM decompression) piped through
    **awk** (C, tag extraction) into **polars** (Rust, classification and
    aggregation).  No Python per-record overhead.

    Returns
    -------
    per_cell_df : polars.DataFrame
        Columns: ``[cb, total_reads, uniquely_mapped, has_secondary,
        has_supplementary, unmapped, dup_reads, unique_umis]``.
    umi_pairs_df : polars.DataFrame
        ``(cb, umi)`` pairs for **non-duplicate** reads (for saturation curve).
    has_dedup : bool
        Whether the BAM contained DT tags (i.e. was dup-marked).
    """
    import shlex
    import subprocess

    logger.info(f"Scanning BAM for per-cell QC stats: {bam_path}")

    # ── Step 1: Extract 5 fields per alignment via samtools + awk ─────────
    awk_prog = (
        'BEGIN{OFS="\\t"}'
        "{"
        '  cb=""; ub=""; dt="";'
        "  for(i=12;i<=NF;i++){"
        '    t=substr($i,1,5);'
        '    if(t=="CB:Z:") cb=substr($i,6);'
        '    else if(t=="UB:Z:") ub=substr($i,6);'
        '    else if(t=="DT:Z:") dt=substr($i,6)'
        "  };"
        "  print $1,$2,cb,ub,dt"
        "}"
    )
    sam_threads = max(1, threads - 1)
    cmd = (
        f"samtools view -@ {sam_threads} {shlex.quote(str(bam_path))}"
        f" | awk -F'\\t' '{awk_prog}'"
    )
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    # ── Step 2: Load into polars from pipe ────────────────────────────────
    df = pl.read_csv(
        proc.stdout,
        separator="\t",
        has_header=False,
        new_columns=["qname", "flag", "cb", "umi", "dt"],
        schema_overrides={"flag": pl.UInt16},
        null_values=[""],
    )
    stderr_out = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(f"samtools/awk failed: {stderr_out.decode()}")

    n_aln = df.height
    logger.info(f"  Loaded {n_aln:,} alignments")

    # ── Step 3: Build sec/supp qname sets, keep only primaries ────────────
    FLAG_SEC = 0x100
    FLAG_SUPP = 0x800
    FLAG_UNMAP = 0x4
    FLAG_DUP = 0x400

    sec_qnames = (
        df.filter((pl.col("flag") & FLAG_SEC) > 0)
        .select("qname").unique()
    )
    supp_only_qnames = (
        df.filter((pl.col("flag") & FLAG_SUPP) > 0)
        .select("qname").unique()
        .join(sec_qnames, on="qname", how="anti")
    )

    primary = df.filter(
        ((pl.col("flag") & (FLAG_SEC | FLAG_SUPP)) == 0)
        & pl.col("cb").is_not_null()
    )
    del df

    # ── Step 4: Classify each primary alignment ──────────────────────────
    has_dedup = primary.filter(pl.col("dt").is_not_null()).height > 0

    primary = (
        primary
        .join(
            sec_qnames.with_columns(pl.lit(True).alias("_sec")),
            on="qname", how="left",
        )
        .join(
            supp_only_qnames.with_columns(pl.lit(True).alias("_supp")),
            on="qname", how="left",
        )
        .with_columns(
            pl.col("_sec").fill_null(False),
            pl.col("_supp").fill_null(False),
            ((pl.col("flag") & FLAG_UNMAP) > 0).alias("is_unmapped"),
            (
                (pl.col("dt") == "Yes")
                | ((pl.col("flag") & FLAG_DUP) > 0)
            ).alias("is_dup"),
        )
        .with_columns(
            pl.when(pl.col("is_unmapped")).then(pl.lit("unmapped"))
              .when(pl.col("_sec")).then(pl.lit("has_secondary"))
              .when(pl.col("_supp")).then(pl.lit("has_supplementary"))
              .otherwise(pl.lit("uniquely_mapped"))
              .alias("aln_class"),
        )
    )
    del sec_qnames, supp_only_qnames

    # ── Step 5: Per-cell aggregation ──────────────────────────────────────
    per_cell_df = primary.group_by("cb").agg(
        pl.len().alias("total_reads"),
        (pl.col("aln_class") == "uniquely_mapped").sum().alias("uniquely_mapped"),
        (pl.col("aln_class") == "has_secondary").sum().alias("has_secondary"),
        (pl.col("aln_class") == "has_supplementary").sum().alias("has_supplementary"),
        (pl.col("aln_class") == "unmapped").sum().alias("unmapped"),
        pl.col("is_dup").sum().alias("dup_reads"),
        pl.col("umi")
          .filter(~pl.col("is_dup") & pl.col("umi").is_not_null())
          .n_unique()
          .alias("unique_umis"),
    )

    # ── Step 6: umi_pairs DataFrame for saturation curve ─────────────────
    umi_pairs_df = primary.filter(
        ~pl.col("is_dup") & pl.col("umi").is_not_null()
    ).select("cb", "umi")

    logger.info(
        f"  {per_cell_df.height:,} cells, "
        f"{per_cell_df['total_reads'].sum():,} primary reads, "
        f"{umi_pairs_df.height:,} non-dup UMI pairs "
        f"(dedup tags: {has_dedup})"
    )

    return per_cell_df, umi_pairs_df, has_dedup


def _plot_saturation_curve(umi_pairs_df, n_subsamples=10, seed=42):
    """
    Sequencing saturation curve from dup-marked BAM data.

    Shuffles once (numpy permutation), then takes cumulative slices to
    count median unique UMIs per cell at each read fraction.

    Returns (title, fig, caption, caption_y) or None.
    """
    if umi_pairs_df is None or umi_pairs_df.is_empty():
        return None

    n_total = umi_pairs_df.height
    fractions = [i / n_subsamples for i in range(1, n_subsamples + 1)]

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)

    medians = []
    for frac in fractions:
        k = min(max(1, int(n_total * frac)), n_total)
        subset = umi_pairs_df[indices[:k]]
        per_cell = subset.group_by("cb").agg(
            pl.col("umi").n_unique().alias("n_umis")
        )
        medians.append(per_cell["n_umis"].median() or 0)
        logger.info(f"  Saturation curve: {frac*100:.0f}% ({k:,} reads)")

    fig = go.Figure(go.Scatter(
        x=[f * 100 for f in fractions],
        y=medians,
        mode="lines+markers",
        line=dict(color="#4C78A8", width=2),
        marker=dict(size=6, color="#4C78A8"),
        hovertemplate="Reads: %{x:.0f}%<br>Median UMIs/cell: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="Fraction of Reads (%)", tickfont_size=13, gridcolor="#f0f0f0"),
        yaxis=dict(title="Median Unique UMIs per Cell", tickfont_size=13,
                   gridcolor="#f0f0f0", zerolinecolor="#ddd"),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
    )

    caption = (
        f"Sequencing saturation: median unique UMIs per cell at increasing "
        f"read fractions (from {n_total:,} non-duplicate reads). "
        "A plateau indicates the library is saturated; a rising curve suggests "
        "additional sequencing would discover new molecules."
    )
    return "Sequencing Saturation", fig, caption, -0.14


def _plot_alignment_stats(per_cell_df):
    """
    Bar chart of global alignment statistics.

    Shows total reads, aligned, uniquely aligned, with secondary
    alignments, with supplementary alignments, and unmapped.

    Returns (title, fig, caption, caption_y) or None.
    """
    if per_cell_df is None or per_cell_df.is_empty():
        return None

    for col in ("total_reads", "uniquely_mapped", "has_secondary", "has_supplementary", "unmapped"):
        if col not in per_cell_df.columns:
            return None

    g_total = per_cell_df["total_reads"].sum()
    g_unique = per_cell_df["uniquely_mapped"].sum()
    g_sec = per_cell_df["has_secondary"].sum()
    g_supp = per_cell_df["has_supplementary"].sum()
    g_unmap = per_cell_df["unmapped"].sum()
    g_aligned = g_total - g_unmap

    def _pct(n):
        return f"{100 * n / g_total:.1f}%" if g_total > 0 else "0%"

    labels = ["Total", "Aligned", "Uniquely\naligned", "W/ secondary", "W/ supplementary", "Unmapped"]
    counts = [g_total, g_aligned, g_unique, g_sec, g_supp, g_unmap]
    colors = ["#4C78A8", "#54A24B", "#72B7B2", "#F58518", "#EECA3B", "#E45756"]
    text = [
        f"{g_total:,}",
        f"{g_aligned:,}<br>({_pct(g_aligned)})",
        f"{g_unique:,}<br>({_pct(g_unique)})",
        f"{g_sec:,}<br>({_pct(g_sec)})",
        f"{g_supp:,}<br>({_pct(g_supp)})",
        f"{g_unmap:,}<br>({_pct(g_unmap)})",
    ]

    fig = go.Figure(go.Bar(
        x=labels, y=counts, text=text,
        textposition="outside", cliponaxis=False,
        textfont=dict(size=13),
        marker=dict(color=colors, line=dict(color="white", width=1)),
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickfont_size=13),
        yaxis=dict(title="Read Count", gridcolor="#f0f0f0", zerolinecolor="#ddd"),
        plot_bgcolor="white", paper_bgcolor="white",
        uniformtext_minsize=13, uniformtext_mode="show",
    )

    caption = (
        f"{g_total:,} total reads. "
        f"{_pct(g_aligned)} aligned ({g_aligned:,}), "
        f"{_pct(g_unique)} uniquely ({g_unique:,}), "
        f"{_pct(g_sec)} with secondary ({g_sec:,}), "
        f"{_pct(g_supp)} with supplementary ({g_supp:,}), "
        f"{_pct(g_unmap)} unmapped ({g_unmap:,}). "
        "Reads with secondary alignments may also have supplementary alignments."
    )
    return "Alignment Statistics", fig, caption, -0.14


def _plot_global_dup_stats(bam_path):
    """
    Bar chart of global duplication statistics from the dedup stats TSV.

    Reads ``<bam_path>.replace('.bam', '_stats.tsv')`` which contains
    Unique Reads and Duplicate Reads counts written by the dedup command.

    Returns (title, fig, caption, caption_y) or None.
    """
    stats_tsv = bam_path.replace(".bam", "_stats.tsv")
    if not os.path.isfile(stats_tsv):
        return None

    metrics = {}
    with open(stats_tsv) as fh:
        next(fh)  # skip header
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                metrics[parts[0]] = int(parts[1])

    uniq = metrics.get("Unique Reads", 0)
    dups = metrics.get("Duplicate Reads", 0)
    total = uniq + dups
    if total == 0:
        return None

    def _pct(n):
        return f"{100 * n / total:.1f}%" if total > 0 else "0%"

    labels = ["Total", "Unique", "Duplicate"]
    counts = [total, uniq, dups]
    colors = ["#4C78A8", "#54A24B", "#E45756"]
    text = [
        f"{total:,}",
        f"{uniq:,}<br>({_pct(uniq)})",
        f"{dups:,}<br>({_pct(dups)})",
    ]

    fig = go.Figure(go.Bar(
        x=labels, y=counts, text=text,
        textposition="outside", cliponaxis=False,
        textfont=dict(size=13),
        marker=dict(color=colors, line=dict(color="white", width=1)),
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickfont_size=13),
        yaxis=dict(title="Read Count", gridcolor="#f0f0f0", zerolinecolor="#ddd"),
        plot_bgcolor="white", paper_bgcolor="white",
        uniformtext_minsize=13, uniformtext_mode="show",
    )

    caption = (
        f"{total:,} total reads. "
        f"{_pct(uniq)} unique ({uniq:,}), "
        f"{_pct(dups)} duplicates ({dups:,})."
    )
    return "Duplication Statistics", fig, caption, -0.14


def _plot_mapping_rate_per_cell(per_cell_df):
    """
    Stacked area chart of alignment categories per cell.

    Shows the fraction of reads per cell that are uniquely mapped,
    have secondary alignments, have supplementary alignments, or are
    unmapped.  Categories are mutually exclusive: secondary takes
    precedence over supplementary (a read with both is counted under
    secondary).  Cells are sorted by total read count (descending).

    Returns (title, fig, caption, caption_y) or None.
    """
    if per_cell_df is None or per_cell_df.is_empty():
        return None

    for col in ("uniquely_mapped", "has_secondary", "has_supplementary", "unmapped", "total_reads"):
        if col not in per_cell_df.columns:
            return None

    df = per_cell_df.sort("total_reads", descending=True)
    n_cells = len(df)
    x = list(range(1, n_cells + 1))

    totals = df["total_reads"].to_list()
    unique = df["uniquely_mapped"].to_list()
    sec = df["has_secondary"].to_list()
    supp = df["has_supplementary"].to_list()
    unmap = df["unmapped"].to_list()

    def _frac(nums, denoms):
        return [n / d if d > 0 else 0 for n, d in zip(nums, denoms)]

    _COLORS = {
        "Uniquely mapped": "#54A24B",
        "W/ secondary": "#F58518",
        "W/ supplementary": "#EECA3B",
        "Unmapped": "#E45756",
    }

    fig = go.Figure()
    for name, vals in [
        ("Uniquely mapped", _frac(unique, totals)),
        ("W/ secondary", _frac(sec, totals)),
        ("W/ supplementary", _frac(supp, totals)),
        ("Unmapped", _frac(unmap, totals)),
    ]:
        fig.add_trace(go.Scatter(
            x=x, y=vals,
            mode="lines",
            name=name,
            stackgroup="one",
            line=dict(width=0.5, color=_COLORS[name]),
            fillcolor=_COLORS[name],
            hovertemplate=f"<b>{name}</b><br>Cell rank: %{{x:,}}<br>Fraction: %{{y:.2%}}<extra></extra>",
        ))

    fig.update_layout(
        xaxis=dict(title=f"Cells (ranked by total reads, n={n_cells:,})",
                   tickfont_size=13, gridcolor="#f0f0f0"),
        yaxis=dict(title="Fraction of Reads", tickfont_size=13, gridcolor="#f0f0f0",
                   zerolinecolor="#ddd", range=[0, 1]),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=True,
        legend=dict(font_size=12, bgcolor="rgba(255,255,255,0.8)"),
    )

    # Global stats for caption
    g_total = sum(totals)
    g_unique = sum(unique)
    g_sec = sum(sec)
    g_supp = sum(supp)
    g_unmap = sum(unmap)

    def _gpct(n):
        return f"{100 * n / g_total:.1f}%" if g_total > 0 else "0%"

    caption = (
        f"Per-cell read alignment breakdown ({n_cells:,} cells, {g_total:,} total reads). "
        f"Global: {_gpct(g_unique)} uniquely mapped, "
        f"{_gpct(g_sec)} with secondary, "
        f"{_gpct(g_supp)} with supplementary, "
        f"{_gpct(g_unmap)} unmapped. "
        "Cells are sorted by total read count (descending)."
    )
    return "Mapping Rate per Cell", fig, caption, -0.14


def _plot_dup_rate_per_cell(per_cell_df, has_dedup):
    """
    Scatter plot: total reads per cell (x) vs duplicate fraction (y).

    Only rendered when the BAM contains dedup tags (DT).

    Returns (title, fig, caption, caption_y) or None.
    """
    if not has_dedup:
        return None
    if per_cell_df is None or per_cell_df.is_empty():
        return None
    for col in ("total_reads", "dup_reads"):
        if col not in per_cell_df.columns:
            return None

    df = per_cell_df.filter(pl.col("total_reads") > 0)
    if df.is_empty():
        return None

    x = df["total_reads"].to_list()
    y = [(d / t) if t > 0 else 0 for d, t in zip(df["dup_reads"].to_list(), x)]
    cbs = df["cb"].to_list()

    fig = go.Figure(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(size=5, color="#E45756", opacity=0.6),
        text=cbs,
        hovertemplate="<b>Cell %{text}</b><br>Reads: %{x:,}<br>Dup rate: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="Total Reads per Cell", type="log", tickfont_size=13, gridcolor="#f0f0f0"),
        yaxis=dict(title="Duplicate Fraction", tickfont_size=13, gridcolor="#f0f0f0",
                   zerolinecolor="#ddd", range=[0, 1]),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
    )

    g_total = sum(x)
    g_dups = per_cell_df["dup_reads"].sum()

    def _gpct(n):
        return f"{100 * n / g_total:.1f}%" if g_total > 0 else "0%"

    caption = (
        f"Per-cell duplicate rate ({len(x):,} cells). "
        f"Global duplicate rate: {_gpct(g_dups)} ({g_dups:,} / {g_total:,}). "
        "X-axis is log-scaled."
    )
    return "Duplicate Rate per Cell", fig, caption, -0.14
