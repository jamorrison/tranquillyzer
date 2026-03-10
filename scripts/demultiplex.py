import logging
import os
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)


def assign_cell_id(row, whitelist_df, barcode_columns):
    if len(barcode_columns) == 1:
        barcode_type = barcode_columns[0]
        corrected_sequences = row[f"corrected_{barcode_type}"].split(",")

        # Match against the whitelist
        matches = whitelist_df[whitelist_df[barcode_type].isin(corrected_sequences)]

        # Prepare result and counters
        match_type_counter = defaultdict(int)
        cell_id_counter = defaultdict(int)

        if len(matches) == 1:  # One exact match
            match_type_counter[f"Exact match ({barcode_type})"] += 1
            cell_id = matches.index[0] + 1  # 1-based indexing
            cell_id_counter[str(cell_id)] += 1
            return cell_id, match_type_counter, cell_id_counter
        elif len(matches) > 1:  # Multiple matches
            match_type_counter[f"Ambiguous match ({barcode_type})"] += 1
            return "ambiguous", match_type_counter, cell_id_counter
        else:  # No match
            match_type_counter[f"No match ({barcode_type})"] += 1
            return "ambiguous", match_type_counter, cell_id_counter

    # Generalized multi-barcode matching based on the model-defined barcode columns.
    corrected_values = {}
    for barcode_type in barcode_columns:
        corrected_key = f"corrected_{barcode_type}"
        corrected_raw = row.get(corrected_key, "")
        corrected_values[barcode_type] = {
            token.strip() for token in str(corrected_raw).split(",") if token and token.strip() and token.strip() != "NMF"
        }

    combinations = list(product(*(sorted(corrected_values[barcode]) for barcode in barcode_columns)))
    matched_cells_by_count = defaultdict(set)

    for combination in combinations:
        mask = pd.Series(True, index=whitelist_df.index)
        for barcode_type, candidate in zip(barcode_columns, combination):
            mask &= whitelist_df[barcode_type] == candidate

        for idx in whitelist_df[mask].index:
            matched_cells_by_count[len(barcode_columns)].add(idx + 1)

        for match_count in range(len(barcode_columns) - 1, 0, -1):
            for matched_columns in product([True, False], repeat=len(barcode_columns)):
                if sum(matched_columns) != match_count:
                    continue
                partial_mask = pd.Series(True, index=whitelist_df.index)
                for include_col, barcode_type, candidate in zip(matched_columns, barcode_columns, combination):
                    if include_col:
                        partial_mask &= whitelist_df[barcode_type] == candidate
                for idx in whitelist_df[partial_mask].index:
                    matched_cells_by_count[match_count].add(idx + 1)

    match_type_counter = defaultdict(int)
    cell_id_counter = defaultdict(int)

    best_match_count = max((count for count, cells in matched_cells_by_count.items() if cells), default=0)
    if best_match_count == 0:
        match_type_counter["Ambiguous"] += 1
        return "ambiguous", match_type_counter, cell_id_counter

    best_cells = matched_cells_by_count[best_match_count]
    if len(best_cells) != 1:
        match_type_counter["Ambiguous"] += 1
        return "ambiguous", match_type_counter, cell_id_counter

    cell_id = next(iter(best_cells))
    if best_match_count == len(barcode_columns):
        label = f"Exact match ({' + '.join(barcode_columns)})"
    else:
        label = f"{best_match_count} of {len(barcode_columns)} match"
    match_type_counter[label] += 1
    cell_id_counter[str(cell_id)] += 1
    return cell_id, match_type_counter, cell_id_counter


def generate_demux_stats_pdf(
    pdf_output_file,
    match_tsv_file,
    cell_tsv_file,
    match_type_counter,
    cell_id_counter,
    cdf_tsv_file=None,
    elbow_tsv_file=None,
):
    plt.style.use("seaborn-v0_8-whitegrid")
    palette = ["#1F78B4", "#33A02C", "#FF7F00", "#E31A1C", "#6A3D9A", "#A6CEE3"]
    grid_color = "#DDE3EA"
    output_dir = os.path.dirname(os.path.abspath(pdf_output_file))
    cdf_tsv_file = cdf_tsv_file or os.path.join(output_dir, "reads_per_cell_cdf.tsv")
    elbow_tsv_file = elbow_tsv_file or os.path.join(output_dir, "reads_per_cell_elbow.tsv")

    with PdfPages(pdf_output_file) as pdf:
        # Save match statistics to TSV
        match_df = pd.DataFrame(match_type_counter.items(), columns=["Match_Type", "Read_Count"])
        match_df.to_csv(match_tsv_file, sep="\t", index=False)

        logger.info("Converted match_type_counter to tsv file")

        # Save Cell ID and Read Counts to TSV
        if cell_id_counter:
            cell_id_data = {int(k): v for k, v in cell_id_counter.items() if k != "ambiguous"}
            cell_df = pd.DataFrame(cell_id_data.items(), columns=["Cell_ID", "Read_Count"])
            cell_df.to_csv(cell_tsv_file, sep="\t", index=False)

        logger.info("Converted cell_id_counter to tsv file")

        # Match Type Bar Plot
        predefined_order = ["Exact match (i5 + i7 + CBC)", "Two out of three match", "Only CBC match", "Ambiguous"]
        match_categories = [category for category in predefined_order if category in match_type_counter]
        if not match_categories:
            match_categories = list(match_type_counter.keys())
        match_counts = [match_type_counter[category] for category in match_categories]

        logger.info("Prepared match type counts for plotting")

        if match_categories:
            fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
            bar_colors = [palette[i % len(palette)] for i in range(len(match_categories))]
            bars = ax.bar(match_categories, match_counts, color=bar_colors, edgecolor="white", linewidth=0.9)
            ax.set_title("Reads per Match Category", fontsize=13, fontweight="semibold", pad=10)
            ax.set_xlabel("Match Category")
            ax.set_ylabel("Read Count")
            ax.set_xticklabels(match_categories, rotation=30, ha="right")
            ax.grid(axis="y", color=grid_color, linewidth=0.8)
            ax.grid(axis="x", visible=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.bar_label(bars, padding=2, fontsize=8, color="#2f2f2f")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        else:
            print("No match types to plot.")

        logger.info("Plotted match type counts")

        # CDF Plot for Reads Per Cell
        if cell_id_counter:
            read_counts = np.array(list(cell_id_data.values()))
            if len(read_counts) == 0:
                logger.info("No non-ambiguous cells to plot in CDF")
                pd.DataFrame(columns=["rank", "reads_per_cell", "cumulative_fraction"]).to_csv(
                    cdf_tsv_file, sep="\t", index=False
                )
                pd.DataFrame(
                    columns=[
                        "elbow_rank",
                        "elbow_reads_per_cell",
                        "elbow_cumulative_fraction",
                        "total_cells",
                        "method",
                    ]
                ).to_csv(elbow_tsv_file, sep="\t", index=False)
                return

            # Sort read counts high->low so the curve starts from the highest-depth cells.
            read_counts_sorted = np.sort(read_counts)[::-1]
            cdf = np.arange(1, len(read_counts_sorted) + 1) / len(read_counts_sorted)
            cdf_df = pd.DataFrame(
                {
                    "rank": np.arange(1, len(read_counts_sorted) + 1),
                    "reads_per_cell": read_counts_sorted,
                    "cumulative_fraction": cdf,
                }
            )
            cdf_df.to_csv(cdf_tsv_file, sep="\t", index=False)

            # Elbow by maximum perpendicular distance from the chord joining start/end points.
            if len(read_counts_sorted) >= 3 and read_counts_sorted[-1] != read_counts_sorted[0]:
                x_norm = (read_counts_sorted - read_counts_sorted[0]) / (read_counts_sorted[-1] - read_counts_sorted[0])
                y_norm = cdf
                x1, y1 = x_norm[0], y_norm[0]
                x2, y2 = x_norm[-1], y_norm[-1]
                denom = np.hypot(y2 - y1, x2 - x1)
                distances = np.abs((y2 - y1) * x_norm - (x2 - x1) * y_norm + x2 * y1 - y2 * x1) / denom
                elbow_idx = int(np.argmax(distances))
            else:
                elbow_idx = int(len(read_counts_sorted) * 0.05)

            elbow_x = read_counts_sorted[elbow_idx]
            elbow_y = cdf[elbow_idx]
            pd.DataFrame(
                [
                    {
                        "elbow_rank": elbow_idx + 1,
                        "elbow_reads_per_cell": elbow_x,
                        "elbow_cumulative_fraction": elbow_y,
                        "total_cells": len(read_counts_sorted),
                        "method": "max_perpendicular_distance_to_chord",
                    }
                ]
            ).to_csv(elbow_tsv_file, sep="\t", index=False)

            fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
            ax.plot(read_counts_sorted, cdf, color="#2A6F97", linewidth=2.2, alpha=0.95)
            ax.fill_between(read_counts_sorted, cdf, color="#2A6F97", alpha=0.08)
            ax.set_xlabel("Reads per Cell (sorted high to low)")
            ax.set_ylabel("Cumulative Fraction")
            ax.set_title("CDF of Reads per Cell", fontsize=13, fontweight="semibold", pad=10)
            ax.axvline(x=elbow_x, color="#D62828", linestyle="--", linewidth=1.6, label=f"Elbow: {elbow_x}")
            ax.axhline(y=elbow_y, color="#D62828", linestyle=":", linewidth=1.2, alpha=0.85)
            ax.scatter([elbow_x], [elbow_y], color="#D62828", s=32, zorder=3)
            ax.annotate(
                f"({elbow_x}, {elbow_y:.2f})",
                xy=(elbow_x, elbow_y),
                xytext=(8, -14),
                textcoords="offset points",
                color="#444444",
                fontsize=9,
            )
            ax.grid(axis="y", color=grid_color, linewidth=0.8)
            ax.grid(axis="x", visible=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        logger.info("Plotted CDF")
