"""
Helper functions for the QC metrics HTML report.

I/O utilities, schema helpers, Polars filter expressions,
Plotly figure builders, and individual metric/plot functions.
All functions are verbatim from qc_metrics_wrap.py — only moved here.
"""

import os
import re

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def _scan_cols(path, cols):
    """
    Lazy-load only the requested columns from a parquet file and collect.
    Columns absent from the file are silently skipped.
    """
    available = _probe_schema(path)
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


# ─────────────────────────────────────────────────────────────────────────────
# report builder  (pure Plotly — no HTML/CSS)
# ─────────────────────────────────────────────────────────────────────────────


_TAG_RE = re.compile(r"<[^>]+>|&[a-z]+;")


def _wrap_caption(text, col_span, n_cols, full_chars=105):
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
    """Unpack a row item as (title, fig, caption, caption_y).

    Accepts 3-tuples ``(title, fig, caption)`` — caption_y defaults to -0.12 —
    or 4-tuples ``(title, fig, caption, caption_y)`` with a per-plot override.
    """
    if len(item) == 4:
        return item
    return (*item, -0.12)


def _build_row_figure(row, n_cols):
    """
    Build a ``go.Figure`` for a single row of subplots using ``make_subplots``.
    Each row figure has its own Plotly modebar with independent zoom/pan/autoscale.

    Parameters
    ----------
    row : list of ``(title, go.Figure, caption[, caption_y])`` tuples
    n_cols : int
        Global column count so caption word-wrap matches across rows.

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
    )

    for col_i, item in enumerate(row):
        _, fig, caption, caption_y = _row_item(item)
        c   = start_cols[col_i]
        sfx = "" if col_i == 0 else str(col_i + 1)

        for trace in fig.data:
            combined.add_trace(trace, row=1, col=c)

        combined.update_xaxes(fig.layout.xaxis.to_plotly_json(), row=1, col=c)
        combined.update_yaxes(fig.layout.yaxis.to_plotly_json(), row=1, col=c)

        if caption:
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

    # Propagate updatemenus and extra annotations from individual figures.
    extra_menus = []
    extra_annots = []
    for item in row:
        _, fig, _, _ = _row_item(item)
        if fig.layout.updatemenus:
            extra_menus.extend(um.to_plotly_json() for um in fig.layout.updatemenus)
        if fig.layout.annotations:
            extra_annots.extend(an.to_plotly_json() for an in fig.layout.annotations)

    combined.update_layout(
        font=dict(size=13),
        height=480,
        bargap=0.55,
        plot_bgcolor="white",
        paper_bgcolor="#f5f7fa",
        showlegend=True,
        legend=dict(font_size=12, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=80 if extra_menus else 50, b=130, l=60, r=40),
        **({"updatemenus": extra_menus} if extra_menus else {}),
    )
    if extra_annots:
        for an in extra_annots:
            combined.add_annotation(**an)
    return combined


def _write_html_report(path, row_figs, sample_name):
    """
    Combine per-row Plotly figures into a single self-contained HTML file.
    Plotly.js is loaded once via CDN; each figure renders with its own modebar.
    """
    import plotly.io as pio

    html_figs = []
    for i, fig in enumerate(row_figs):
        html_figs.append(pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs="cdn" if i == 0 else False,
            config={"displaylogo": False},
        ))

    page_title = f"Tranquillyzer QC Report \u2014 {sample_name}"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "<!DOCTYPE html><html>"
            "<head><meta charset='utf-8'><style>"
            "body{background:#f5f7fa;font-family:Arial,sans-serif;margin:0;padding:16px}"
            "h1{text-align:center;font-size:18px;color:#333;margin:0 0 8px}"
            "</style></head>"
            f"<body><h1>{page_title}</h1>"
        )
        fh.write("\n".join(html_figs))
        fh.write("</body></html>")


# ─────────────────────────────────────────────────────────────────────────────
# metric / plot functions
# ─────────────────────────────────────────────────────────────────────────────


def _compute_summary(valid_path, invalid_path, vcols):
    """
    Compute total / valid / invalid / demuxed / ambiguous counts.
    Needs: valid → [cell_id]; invalid → row count only.
    Returns a plain dict of counts.
    """
    n_valid   = _count_rows(valid_path)
    n_invalid = _count_rows(invalid_path)
    n_total   = n_valid + n_invalid

    cell_col = _first_present(vcols, ["cell_id", "corrected_CBC"])
    has_cell = cell_col is not None
    n_demuxed = n_ambiguous = 0
    if has_cell:
        cell_df     = _scan_cols(valid_path, [cell_col])
        n_demuxed   = cell_df.filter(_expr_is_demuxed(cell_col)).height
        n_ambiguous = cell_df.filter(_expr_is_ambiguous(cell_col)).height

    return {
        "n_total":     n_total,
        "n_valid":     n_valid,
        "n_invalid":   n_invalid,
        "n_demuxed":   n_demuxed,
        "n_ambiguous": n_ambiguous,
        "has_cell":    has_cell,
        "cell_col":    cell_col,
    }


def _plot_read_architecture(summary, sample_name):
    """
    Plot 1 — Read Architecture: Total / Valid / Invalid bars.
    Returns (title, go.Figure, caption).
    """
    n_total, n_valid, n_invalid = summary["n_total"], summary["n_valid"], summary["n_invalid"]

    def _pct(n, d):
        return 100.0 * n / d if d > 0 else 0.0

    labels = ["Total", "Valid", "Invalid"]
    colors = ["#4C78A8", "#54A24B", "#E45756"]
    counts = [n_total, n_valid, n_invalid]
    text   = [
        f"{n_total:,}",
        f"{n_valid:,}<br>({_pct(n_valid, n_total):.1f}%)",
        f"{n_invalid:,}<br>({_pct(n_invalid, n_total):.1f}%)",
    ]

    fig = go.Figure(go.Bar(
        x=labels, y=counts, text=text,
        textposition="outside", cliponaxis=False,
        textfont=dict(size=13),
        marker=dict(color=colors, line=dict(color="white", width=1.5)),
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickfont_size=13),
        yaxis=dict(tickfont_size=13, title="Read Count", gridcolor="#f0f0f0", zerolinecolor="#ddd"),
        plot_bgcolor="white", paper_bgcolor="white",
    )

    caption = (
        "Reads classified by predicted segment order vs. expected library structure."
        "\n\n"
        "<b>Valid</b>: all segments in correct order.&nbsp;&nbsp;"
        "<b>Invalid</b>: unexpected, missing, or out-of-order segments."
        "\n\n"
    )
    return "Read Architecture", fig, caption, -0.14


def _plot_barcode_assignment(summary, sample_name):
    """
    Plot 2 — Barcode Assignment: Demuxed / Ambiguous bars (valid reads only).
    Returns (title, go.Figure, caption), or None when no cell assignment data.
    """
    if not summary["has_cell"]:
        return None

    n_valid, n_demuxed, n_ambiguous = (
        summary["n_valid"], summary["n_demuxed"], summary["n_ambiguous"]
    )

    def _pct(n, d):
        return 100.0 * n / d if d > 0 else 0.0

    labels = ["Demuxed", "Ambiguous"]
    colors = ["#72B7B2", "#F58518"]
    counts = [n_demuxed, n_ambiguous]
    text   = [
        f"{n_demuxed:,}<br>({_pct(n_demuxed, n_valid):.1f}%)",
        f"{n_ambiguous:,}<br>({_pct(n_ambiguous, n_valid):.1f}%)",
    ]

    fig = go.Figure(go.Bar(
        x=labels, y=counts, text=text,
        textposition="outside", cliponaxis=False,
        textfont=dict(size=13),
        marker=dict(color=colors, line=dict(color="white", width=1.5)),
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickfont_size=13),
        yaxis=dict(title="Read Count", gridcolor="#f0f0f0", zerolinecolor="#ddd"),
        plot_bgcolor="white", paper_bgcolor="white",
    )

    caption = (
        "From valid reads, each is assigned to a cell barcode via Levenshtein distance."
        "\n\n"
    )
    return "Barcode Assignment (valid reads)", fig, caption, -0.14


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

    df = _scan_cols(invalid_path, ["reason"]).drop_nulls()
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

    def _label(r):
        return r if len(r) <= _MAX_LABEL else r[:_MAX_LABEL - 1] + "…"

    labels  = [_label(r) for r in all_reasons]
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
            y=1.27, yanchor="top",
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

        df = _scan_cols(valid_path, [dist_col]).drop_nulls()
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
            marker=dict(color=_BAR_COLOR, line=dict(color="white", width=1.5)),
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


def _plot_read_length_dist(valid_path, invalid_path, vcols, bin_width):
    """
    Line chart of read-length distribution for five read subsets.
    Provides interactive dropdowns to change bin width and cap the x-axis.
    Returns (title, go.Figure, caption) or None.
    """
    if "read_length" not in vcols:
        return None

    _COLORS = {
        "Total":             "#888888",
        "Invalid":           "#E45756",
        "Valid":             "#54A24B",
        "Valid — Demuxed":   "#72B7B2",
        "Valid — Ambiguous": "#F58518",
    }
    _NBIN_PRESETS = [50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000,
                     200_000, 300_000, 400_000, 500_000, 1_000_000]

    cell_col  = _first_present(list(vcols), ["cell_id", "corrected_CBC"])
    load_cols = ["read_length"] + ([cell_col] if cell_col else [])
    valid_df  = _scan_cols(valid_path,   load_cols)       if valid_path   else pl.DataFrame()
    inv_df    = _scan_cols(invalid_path, ["read_length"]) if invalid_path else pl.DataFrame()

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

    def _bin_counts_bw(df, bw):
        return dict(
            df.with_columns(
                (pl.col("read_length") // bw * bw).cast(pl.Int64).alias("bin")
            )
            .group_by("bin").len().sort("bin").rows()
        )

    # ── add all traces (n_groups × n_bins options); only default visible ──────
    fig = go.Figure()
    for nb_idx, n_bins in enumerate(all_nbins):
        bw         = max(1, max_read_len // n_bins)
        all_counts = {nm: _bin_counts_bw(group_dfs[nm], bw) for nm in group_names}
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
            dict(text="Bins:", x=0.0,  xanchor="left", y=1.26, yanchor="top",
                 xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12, color="#444")),
            dict(text="Cap x-axis:", x=0.12, xanchor="left", y=1.26, yanchor="top",
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
        "<b>Valid — Demuxed</b>: unambiguously assigned to a cell.&nbsp;&nbsp;"
        "<b>Valid — Ambiguous</b>: ambiguous cell assignment."
    )
    return "Read-Length Distribution", fig, caption, -0.19


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

    df = _scan_cols(valid_path, [bc_col])
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
        line=dict(color="#4C78A8", width=2),
        name="Reads per barcode",
        hovertemplate="Rank %{x:,}<br>Reads: %{y:,}<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(
            title="Barcodes (rank)",
            type="log",
            tickfont_size=13,
            gridcolor="#f0f0f0",
        ),
        yaxis=dict(
            title="Reads (transcripts proxy)",
            type="log",
            tickfont_size=13,
            gridcolor="#f0f0f0",
            zerolinecolor="#ddd",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )

    caption = (
        f"Barcode knee plot for {len(ranks):,} demuxed cell IDs (ambiguous reads excluded). "
        "Cell IDs are ranked by read count (descending); both axes are log-scaled. "
        "Note: UMI deduplication is applied at the BAM level and is not reflected here."
    )
    return "Transcripts vs Barcodes", fig, caption, -0.18


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
        .with_columns(pl.col(cell_col).cast(pl.Utf8).alias("_cid"))
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

    _BAND_COLOR   = "rgba(114,183,178,0.25)"   # teal, semi-transparent
    _MEDIAN_COLOR = "#1f77b4"

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
