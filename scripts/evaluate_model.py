import logging
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rapidfuzz.distance import Levenshtein

from scripts.extract_annotated_seqs import collapse_labels

logger = logging.getLogger(__name__)


# ─── helpers ───


def _parse_coord_list(val):
    """Parse a comma-separated coordinate string into a list of ints."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    val_str = str(val).strip()
    if val_str == "" or val_str.lower() == "nan":
        return []
    parts = []
    for x in val_str.split(","):
        x = x.strip()
        if x and x.lower() != "nan":
            parts.append(int(float(x)))
    return parts


def _extract_gt_segment_sequences(gt_labels, read_seq, seq_order):
    """Extract GT segment sequences from per-base labels and the read sequence.

    Returns {segment: [seq1, seq2, ...]} for each segment type.
    """
    read_length = len(gt_labels)
    _, _, indices_dict = collapse_labels(gt_labels, read_length)
    result = {}
    for seg in seq_order:
        seqs = []
        for start, end in indices_dict.get(seg, []):
            seqs.append(read_seq[start:end])
        result[seg] = seqs
    return result


def _extract_pred_segment_sequences(row, seg_columns, read_seq):
    """Extract predicted segment sequences from parquet row coordinates + read."""
    result = {}
    for seg, (start_col, end_col) in seg_columns.items():
        starts = _parse_coord_list(row.get(start_col))
        ends = _parse_coord_list(row.get(end_col))
        seqs = []
        for s, e in zip(starts, ends):
            seqs.append(read_seq[s:e])
        result[seg] = seqs
    return result


def _detect_seg_columns(columns, seq_order):
    """Detect {segment: (start_col, end_col)} for segments present in parquet."""
    seg_cols = {}
    for seg in seq_order:
        s_col = f"{seg}_Starts"
        e_col = f"{seg}_Ends"
        if s_col in columns and e_col in columns:
            seg_cols[seg] = (s_col, e_col)
    return seg_cols


def _reconstruct_per_base_labels(row, seg_columns, read_length):
    """Reconstruct per-base label array from parquet segment coordinates."""
    labels = ["unlabeled"] * read_length
    for seg, (start_col, end_col) in seg_columns.items():
        starts = _parse_coord_list(row.get(start_col))
        ends = _parse_coord_list(row.get(end_col))
        for s, e in zip(starts, ends):
            for pos in range(s, min(e, read_length)):
                labels[pos] = seg
    return labels


def _plot_classification_metrics(gt_labels_list, pred_labels_list, seq_order):
    """Compute per-base precision, recall, F1 and return (figure, dataframe).

    Returns a grouped bar chart + an HTML table string for embedding.
    Computes metrics without sklearn to avoid scipy/GLIBCXX dependency issues.
    """
    y_true = [lbl for read_labels in gt_labels_list for lbl in read_labels]
    y_pred = [lbl for read_labels in pred_labels_list for lbl in read_labels]

    # Exclude unlabeled positions
    paired = [(t, p) for t, p in zip(y_true, y_pred) if t != "unlabeled" and p != "unlabeled"]
    if not paired:
        return None, pd.DataFrame(), ""

    y_true_clean, y_pred_clean = zip(*paired)

    all_labels = sorted(set(y_true_clean) | set(y_pred_clean))
    ordered_labels = [s for s in seq_order if s in all_labels]
    ordered_labels += [s for s in all_labels if s not in ordered_labels]

    # Count TP, FP, FN per label
    tp = {lbl: 0 for lbl in ordered_labels}
    fp = {lbl: 0 for lbl in ordered_labels}
    fn = {lbl: 0 for lbl in ordered_labels}
    for t, p in zip(y_true_clean, y_pred_clean):
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[p] += 1

    total_support = len(y_true_clean)

    rows = []
    for label in ordered_labels:
        support = tp[label] + fn[label]
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "segment": label,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "support": support,
            }
        )

    # Macro average
    n_labels = len(ordered_labels)
    macro_p = sum(r["precision"] for r in rows) / n_labels if n_labels else 0
    macro_r = sum(r["recall"] for r in rows) / n_labels if n_labels else 0
    macro_f1 = sum(r["f1_score"] for r in rows) / n_labels if n_labels else 0
    rows.append(
        {
            "segment": "macro avg",
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1_score": round(macro_f1, 4),
            "support": total_support,
        }
    )

    # Weighted average
    weighted_p = sum(r["precision"] * r["support"] for r in rows[:-1]) / total_support if total_support else 0
    weighted_r = sum(r["recall"] * r["support"] for r in rows[:-1]) / total_support if total_support else 0
    weighted_f1 = sum(r["f1_score"] * r["support"] for r in rows[:-1]) / total_support if total_support else 0
    rows.append(
        {
            "segment": "weighted avg",
            "precision": round(weighted_p, 4),
            "recall": round(weighted_r, 4),
            "f1_score": round(weighted_f1, 4),
            "support": total_support,
        }
    )

    df = pd.DataFrame(rows)

    # Bar chart (exclude averages)
    seg_df = df[~df["segment"].isin(["macro avg", "weighted avg"])]
    fig = go.Figure()
    for metric, name in [("precision", "Precision"), ("recall", "Recall"), ("f1_score", "F1 Score")]:
        fig.add_trace(
            go.Bar(
                x=seg_df["segment"],
                y=seg_df[metric],
                name=name,
                marker_color=_PLOT_COLOR,
                opacity=0.6 if metric == "precision" else (0.8 if metric == "recall" else 1.0),
                text=seg_df[metric].apply(lambda v: f"{v:.2f}"),
                textposition="outside",
            )
        )
    fig.update_layout(
        barmode="group",
        height=450,
        yaxis=dict(range=[0, 1.15], title="Score", tickfont_size=13, gridcolor="#f0f0f0"),
        xaxis=dict(tickfont_size=13, gridcolor="#f0f0f0"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # HTML table
    table_rows = ""
    for _, r in df.iterrows():
        seg = r["segment"]
        bold = ' style="font-weight:bold"' if seg in ("macro avg", "weighted avg") else ""
        table_rows += (
            f"<tr{bold}><td>{seg}</td>"
            f"<td>{r['precision']:.4f}</td>"
            f"<td>{r['recall']:.4f}</td>"
            f"<td>{r['f1_score']:.4f}</td>"
            f"<td>{int(r['support']):,}</td></tr>"
        )
    html_table = (
        '<table style="border-collapse:collapse;margin-top:12px;font-size:13px;width:100%">'
        '<thead><tr style="border-bottom:2px solid #333;text-align:left">'
        "<th>Segment</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Support</th>"
        "</tr></thead><tbody>"
        f"{table_rows}"
        "</tbody></table>"
    )

    return fig, df, html_table


def _write_tsv_with_metadata(df, path, model_name, version):
    """Write a DataFrame to TSV with metadata comment header."""
    with open(path, "w") as fh:
        fh.write(f"# tranquillyzer_version: {version}\n")
        fh.write(f"# model_name: {model_name}\n")
        df.to_csv(fh, sep="\t", index=False)


# ─── read-architecture metrics ───


_PLOT_COLOR = "#4A7FB5"


def _plot_architecture_accuracy(expected_fragments, structure_names, valid_df, invalid_df):
    """Compute structural filtering accuracy and fragment recovery per structure type.

    Single-fragment structures: proportion correctly classified as valid.
    Concatenated structures: proportion of sub-fragments recoverable (manuscript Fig 2D style).

    Returns (figure, dataframe).
    """
    # Count predicted valid fragments per base read name
    pred_frags = {}
    if valid_df is not None and len(valid_df) > 0:
        for name in valid_df["ReadName"]:
            base_name = name.split("__frag")[0] if "__frag" in name else name
            pred_frags[base_name] = pred_frags.get(base_name, 0) + 1

    # Group reads by structure type
    groups = {}
    for i, (n_expected, stype) in enumerate(zip(expected_fragments, structure_names)):
        read_name = f"assess_{i}"
        n_predicted = pred_frags.get(read_name, 0)
        if stype not in groups:
            groups[stype] = {
                "n_reads": 0,
                "expected_per_read": n_expected,
                "total_expected_frags": 0,
                "total_predicted_frags": 0,
                "n_correct": 0,
            }
        g = groups[stype]
        g["n_reads"] += 1
        g["total_expected_frags"] += n_expected
        g["total_predicted_frags"] += n_predicted
        if n_expected == 1 and n_predicted >= 1:
            g["n_correct"] += 1
        elif n_expected > 1 and n_predicted == n_expected:
            g["n_correct"] += 1

    # Split into single vs concat
    single_types = {k: v for k, v in groups.items() if v["expected_per_read"] == 1}
    concat_types = {k: v for k, v in groups.items() if v["expected_per_read"] > 1}

    rows = []
    for stype in sorted(groups):
        g = groups[stype]
        if g["expected_per_read"] == 1:
            accuracy = g["n_correct"] / g["n_reads"] if g["n_reads"] > 0 else 0
            recovery = accuracy  # same for single-fragment
        else:
            recovery = g["total_predicted_frags"] / g["total_expected_frags"] if g["total_expected_frags"] > 0 else 0
            accuracy = g["n_correct"] / g["n_reads"] if g["n_reads"] > 0 else 0
        rows.append(
            {
                "structure_type": stype,
                "expected_fragments": g["expected_per_read"],
                "n_reads": g["n_reads"],
                "structural_accuracy": round(accuracy, 4),
                "fragment_recovery_rate": round(recovery, 4),
            }
        )

    df = pd.DataFrame(rows)

    # Build figure
    has_single = len(single_types) > 0
    has_concat = len(concat_types) > 0
    n_cols = has_single + has_concat
    subtitles = []
    if has_single:
        subtitles.append("<b>Structural Filtering Accuracy (Single-Fragment)</b>")
    if has_concat:
        subtitles.append("<b>Sub-Fragment Recovery Rate (Concatenated)</b>")

    fig = make_subplots(rows=1, cols=max(1, n_cols), subplot_titles=subtitles)
    for ann in fig.layout.annotations:
        ann.font = dict(size=15, color="#1a1a2e", family="Arial")

    col = 1
    if has_single:
        single_df = df[df["expected_fragments"] == 1].sort_values("structure_type")
        fig.add_trace(
            go.Bar(
                x=single_df["structure_type"],
                y=single_df["structural_accuracy"],
                marker_color=_PLOT_COLOR,
                text=single_df.apply(lambda r: f"{r['structural_accuracy']:.2f}<br>n={int(r['n_reads'])}", axis=1),
                textposition="outside",
                name="Accuracy",
            ),
            row=1,
            col=col,
        )
        fig.update_yaxes(range=[0, 1.15], row=1, col=col)
        col += 1

    if has_concat:
        concat_df = df[df["expected_fragments"] > 1].sort_values("structure_type")
        fig.add_trace(
            go.Bar(
                x=concat_df["structure_type"],
                y=concat_df["fragment_recovery_rate"],
                marker_color=_PLOT_COLOR,
                text=concat_df.apply(lambda r: f"{r['fragment_recovery_rate']:.2f}<br>n={int(r['n_reads'])}", axis=1),
                textposition="outside",
                name="Recovery",
            ),
            row=1,
            col=col,
        )
        fig.update_yaxes(range=[0, 1.15], row=1, col=col)

    fig.update_layout(
        height=450,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig, df


# ─── segment edit distance metrics ───


def _compute_edit_distances(gt_segs_list, pred_segs_list, seq_order):
    """Compute edit distances between GT and predicted segment sequences.

    Returns a DataFrame with per-read per-segment distances.
    """
    segments = [s for s in seq_order if s not in ("cDNA", "random_s", "random_e")]

    records = []
    for gt_segs, pred_segs in zip(gt_segs_list, pred_segs_list):
        for seg in segments:
            gt_seq_list = gt_segs.get(seg, [])
            pred_seq_list = pred_segs.get(seg, [])

            for k in range(max(len(gt_seq_list), len(pred_seq_list))):
                gt_s = gt_seq_list[k] if k < len(gt_seq_list) else ""
                pr_s = pred_seq_list[k] if k < len(pred_seq_list) else ""
                raw_dist = Levenshtein.distance(gt_s, pr_s)
                gt_len = len(gt_s) if gt_s else 1
                norm_dist = raw_dist / gt_len
                records.append(
                    {
                        "segment": seg,
                        "raw_edit_distance": raw_dist,
                        "normalized_edit_distance": round(norm_dist, 4),
                        "gt_length": len(gt_s),
                        "pred_length": len(pr_s),
                    }
                )

    return pd.DataFrame(records)


# ─── HTML report ───


def _write_html_report(path, panels, model_name):
    """Combine Plotly figures into a single HTML report.

    Parameters
    ----------
    panels : list of (fig, title, caption) tuples
        Each panel has a Plotly figure, a title string, and a caption string.
    """
    import plotly.io as pio
    from utils import get_version

    __version__ = get_version()

    page_title = f"Tranquillyzer v{__version__} Model Assessment \u2014 {model_name}"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "<!DOCTYPE html><html>"
            f"<head><meta charset='utf-8'><meta name='generator' content='tranquillyzer v{__version__}'><style>"
            "body{background:#f5f7fa;font-family:Arial,sans-serif;margin:0;padding:16px}"
            "h1{text-align:center;font-size:18px;color:#333;margin:0 0 8px}"
            ".panel{background:white;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,0.12);"
            "margin-bottom:16px;padding:16px}"
            ".panel-title{font-size:16px;font-weight:bold;color:#1a1a2e;margin:0 0 4px}"
            ".panel-caption{font-size:13px;color:#555;margin:8px 0 0;line-height:1.5;text-align:center}"
            "</style></head>"
            f"<body><h1>{page_title}</h1>"
        )
        for i, (fig, title, caption) in enumerate(panels):
            fig_html = pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn" if i == 0 else False,
                config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            )
            fh.write('<div class="panel">')
            fh.write(f'<div class="panel-title">{title}</div>')
            fh.write(fig_html)
            if caption:
                fh.write(f'<div class="panel-caption">{caption}</div>')
            fh.write("</div>")
        fh.write("</body></html>")


# ─── top-level orchestrator ───


def evaluate_model(
    gt_labels,
    expected_fragments,
    structure_names,
    valid_parquet_path,
    invalid_parquet_path,
    seq_order,
    output_dir,
    model_name,
):
    """Run all assessment metrics and produce an HTML report + TSV artifacts."""
    from utils import get_version

    os.makedirs(output_dir, exist_ok=True)
    __version__ = get_version()

    # Load annotation parquets
    valid_df = pd.read_parquet(valid_parquet_path) if os.path.exists(valid_parquet_path) else pd.DataFrame()
    invalid_df = pd.read_parquet(invalid_parquet_path) if os.path.exists(invalid_parquet_path) else pd.DataFrame()

    panels = []  # list of (fig, title, caption) for HTML report

    # ── 1. Read-architecture accuracy ──
    fig_arch, df_arch = _plot_architecture_accuracy(
        expected_fragments,
        structure_names,
        valid_df,
        invalid_df,
    )
    _write_tsv_with_metadata(
        df_arch, os.path.join(output_dir, f"{model_name}_architecture_accuracy.tsv"), model_name, __version__
    )

    arch_caption = (
        "<b>Structural filtering accuracy</b> (left) shows the fraction of single-fragment reads "
        "correctly classified as valid by the model. "
        "<b>Sub-fragment recovery rate</b> (right) shows the proportion of expected sub-fragments "
        "that the model successfully identified within concatenated reads, "
        "normalized to the known ground truth for each concatenation pattern. "
        "Values closer to 1.0 indicate better performance."
    )
    panels.append((fig_arch, "Read Architecture Assessment", arch_caption))

    for _, row in df_arch.iterrows():
        logger.info(
            f"Structure '{row['structure_type']}': accuracy={row['structural_accuracy']:.3f}, "
            f"recovery={row['fragment_recovery_rate']:.3f} (n={int(row['n_reads'])})"
        )

    # ── 2. Segment edit distance (single-fragment valid reads only) ──
    all_columns = list(valid_df.columns) if len(valid_df) > 0 else []
    seg_columns = _detect_seg_columns(all_columns, seq_order)

    gt_segs_list = []
    pred_segs_list = []
    if len(valid_df) > 0:
        for _, row in valid_df.iterrows():
            name = row["ReadName"]
            if "__frag" in name:
                continue
            try:
                idx = int(name.replace("assess_", ""))
            except (ValueError, AttributeError):
                continue
            read_seq = row.get("read", "")
            gt_segs_list.append(_extract_gt_segment_sequences(gt_labels[idx], read_seq, seq_order))
            pred_segs_list.append(_extract_pred_segment_sequences(row.to_dict(), seg_columns, read_seq))

    if gt_segs_list:
        df_edit = _compute_edit_distances(gt_segs_list, pred_segs_list, seq_order)
        _write_tsv_with_metadata(
            df_edit,
            os.path.join(output_dir, f"{model_name}_segment_edit_distance.tsv"),
            model_name,
            __version__,
        )

        n_reads = len(gt_segs_list)
        segments = [s for s in seq_order if s not in ("cDNA", "random_s", "random_e")]

        # Side-by-side box plots: raw (left) and normalized (right)
        fig_edit = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["<b>Raw Edit Distance (bp)</b>", "<b>Normalized Edit Distance</b>"],
            shared_yaxes=True,
            horizontal_spacing=0.08,
        )
        for ann in fig_edit.layout.annotations:
            ann.font = dict(size=15, color="#1a1a2e", family="Arial")

        ticktext = list(reversed(segments))
        _CAP_H = 0.15
        for i, seg in enumerate(ticktext):
            seg_data = df_edit[df_edit["segment"] == seg]
            if seg_data.empty:
                continue
            for col_idx, value_col, unit_fmt in [
                (1, "raw_edit_distance", ("{:,.0f} bp", "{:,.0f} bp")),
                (2, "normalized_edit_distance", ("{:,.3f}", "{:,.3f}")),
            ]:
                vals = seg_data[value_col].dropna()
                if vals.empty:
                    continue
                n = len(vals)
                q1 = float(vals.quantile(0.25))
                median = float(vals.median())
                q3 = float(vals.quantile(0.75))
                mn, mx = float(vals.min()), float(vals.max())
                iqr = q3 - q1
                lower = max(mn, q1 - 1.5 * iqr)
                upper = min(mx, q3 + 1.5 * iqr)
                vfmt = unit_fmt[0]

                fig_edit.add_trace(
                    go.Bar(
                        y=[i],
                        x=[q3 - q1],
                        base=[q1],
                        orientation="h",
                        width=0.5,
                        marker=dict(
                            color=_PLOT_COLOR,
                            opacity=0.45,
                            line=dict(color=_PLOT_COLOR, width=1.5),
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{seg}</b><br>"
                            f"Median: {vfmt.format(median)}<br>"
                            f"Q1: {vfmt.format(q1)}<br>"
                            f"Q3: {vfmt.format(q3)}<br>"
                            f"Lower fence: {vfmt.format(lower)}<br>"
                            f"Upper fence: {vfmt.format(upper)}<br>"
                            f"n = {n:,}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=col_idx,
                )
                # Whisker lines
                for x0, x1 in [(lower, q1), (q3, upper)]:
                    fig_edit.add_trace(
                        go.Scatter(
                            x=[x0, x1],
                            y=[i, i],
                            mode="lines",
                            line=dict(color=_PLOT_COLOR, width=1.5),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=col_idx,
                    )
                # Whisker caps
                for xc in [lower, upper]:
                    fig_edit.add_trace(
                        go.Scatter(
                            x=[xc, xc],
                            y=[i - _CAP_H, i + _CAP_H],
                            mode="lines",
                            line=dict(color=_PLOT_COLOR, width=1.5),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=col_idx,
                    )
                # Median line
                fig_edit.add_trace(
                    go.Scatter(
                        x=[median, median],
                        y=[i - 0.25, i + 0.25],
                        mode="lines",
                        line=dict(color="black", width=2.5),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col_idx,
                )

        fig_edit.update_layout(
            height=max(350, 50 * len(segments) + 100),
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="closest",
            bargap=0.3,
        )
        fig_edit.update_xaxes(title_text="Edit Distance (bp)", tickfont_size=13, gridcolor="#f0f0f0", row=1, col=1)
        fig_edit.update_xaxes(
            title_text="Normalized Edit Distance", tickfont_size=13, gridcolor="#f0f0f0", row=1, col=2
        )
        for c in (1, 2):
            fig_edit.update_yaxes(
                tickvals=list(range(len(ticktext))),
                ticktext=ticktext,
                tickfont_size=13,
                gridcolor="#f0f0f0",
                row=1,
                col=c,
            )

        edit_caption = (
            f"Levenshtein edit distance between the model's predicted segment sequences and the "
            f"known ground-truth sequences, across {n_reads:,} single-fragment valid reads. "
            f"<b>Left:</b> raw edit distance in base pairs \u2014 lower values indicate more accurate "
            f"boundary detection. <b>Right:</b> normalized edit distance (edit distance \u00f7 ground-truth "
            f"segment length) \u2014 a value of 0 means an exact match; 1 means the edit distance equals "
            f"the full segment length, allowing comparison across segments of different sizes. "
            f"Each box shows the median, IQR, and whiskers (1.5\u00d7 IQR). "
            f"Segments cDNA, random_s, and random_e are excluded."
        )
        panels.append((fig_edit, "Segment Edit Distance", edit_caption))

        for seg in df_edit["segment"].unique():
            seg_data = df_edit[df_edit["segment"] == seg]
            logger.info(
                f"Segment '{seg}': median_edit_dist={seg_data['raw_edit_distance'].median():.0f}bp, "
                f"median_normalized={seg_data['normalized_edit_distance'].median():.4f} "
                f"(n={len(seg_data)})"
            )
        # ── 3. Per-base classification metrics (P/R/F1) ──
        pred_labels_list = []
        gt_labels_matched = []
        for _, row in valid_df.iterrows():
            name = row["ReadName"]
            if "__frag" in name:
                continue
            try:
                idx = int(name.replace("assess_", ""))
            except (ValueError, AttributeError):
                continue
            read_length = int(row.get("read_length", 0))
            if read_length == 0:
                continue
            pred_labels_list.append(_reconstruct_per_base_labels(row.to_dict(), seg_columns, read_length))
            gt_labels_matched.append(gt_labels[idx][:read_length])

        if pred_labels_list:
            result = _plot_classification_metrics(gt_labels_matched, pred_labels_list, seq_order)
            if result[0] is not None:
                fig_cls, df_cls, html_table = result
                _write_tsv_with_metadata(
                    df_cls,
                    os.path.join(output_dir, f"{model_name}_classification_report.tsv"),
                    model_name,
                    __version__,
                )

                cls_caption = (
                    f"Per-base classification metrics across {len(pred_labels_list):,} single-fragment "
                    f"valid reads. Every base position in each read is assigned a segment label by both "
                    f"the ground truth and the model's prediction. <b>Precision</b>: of all bases the model "
                    f"labeled as a given segment, what fraction truly belonged to that segment. "
                    f"<b>Recall</b>: of all bases that truly belonged to a segment, what fraction did the "
                    f"model correctly identify. <b>F1 Score</b>: harmonic mean of precision and recall. "
                    f"<b>Support</b>: total number of ground-truth bases for that segment."
                    f"{html_table}"
                )
                panels.append((fig_cls, "Per-Base Classification Metrics", cls_caption))

                # Log weighted averages
                weighted = df_cls[df_cls["segment"] == "weighted avg"]
                if not weighted.empty:
                    w = weighted.iloc[0]
                    logger.info(f"Weighted avg: P={w['precision']:.4f}, R={w['recall']:.4f}, F1={w['f1_score']:.4f}")

    else:
        logger.warning("No matched single-fragment reads for segment metrics computation")

    # ── Write HTML report ──
    report_path = os.path.join(output_dir, f"{model_name}_assessment_report.html")
    _write_html_report(report_path, panels, model_name)
    logger.info(f"Assessment report saved to {report_path}")
