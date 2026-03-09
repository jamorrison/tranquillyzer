import logging
import gc
import os
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from rapidfuzz import process
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import defaultdict
from rapidfuzz.distance import Levenshtein
from matplotlib.backends.backend_pdf import PdfPages

from scripts.demultiplex import assign_cell_id

logger = logging.getLogger(__name__)


def reverse_complement(seq):
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(complement.get(base, base) for base in reversed(seq))


def correct_barcode(row, column_name, whitelist, threshold):
    observed_barcode = row[column_name]
    reverse_comp_barcode = reverse_complement(observed_barcode)

    # Get distance scores for observed barcode and reverse complement
    candidates = process.extract(observed_barcode, whitelist, scorer=Levenshtein.distance, limit=5)
    candidates_rev = process.extract(reverse_comp_barcode, whitelist, scorer=Levenshtein.distance, limit=5)

    # Combine results and find minimum distance
    all_matches = candidates + candidates_rev
    min_distance = min(match[1] for match in all_matches)

    # Find all closest barcodes with the same minimum distance
    closest_barcodes = [match[0] for match in all_matches if match[1] == min_distance]

    # Handle threshold & multiple matches
    if min_distance > threshold:
        # if len(closest_barcodes) == 1:
        #     return observed_barcode, closest_barcodes[0], min_distance, 1
        return observed_barcode, "NMF", min_distance, len(closest_barcodes)

    return observed_barcode, ",".join(closest_barcodes), min_distance, len(closest_barcodes)


def write_reads_to_fasta(
    batch_reads, output_fmt, demuxed_fasta, demuxed_fasta_lock, ambiguous_fasta, ambiguous_fasta_lock
):
    for cell_id, reads in batch_reads.items():
        if cell_id == "ambiguous":
            if ambiguous_fasta_lock:
                with ambiguous_fasta_lock:
                    fasta_file = open(ambiguous_fasta, "a")
            else:
                fasta_file = open(ambiguous_fasta, "a")
        else:
            if demuxed_fasta_lock:
                with demuxed_fasta_lock:
                    fasta_file = open(demuxed_fasta, "a")
            else:
                fasta_file = open(demuxed_fasta, "a")

        if output_fmt == "fastq":
            for header, sequence, quality in reads:
                fasta_file.write(f"{header}\n{sequence}\n+\n{quality}\n")
        elif output_fmt == "fasta":
            for header, sequence in reads:
                fasta_file.write(f"{header}\n{sequence}\n")

        fasta_file.close()


def process_row(
    row,
    strand,
    barcode_columns,
    whitelist_dict,
    whitelist_df,
    threshold,
    output_dir,
    output_fmt,
    include_barcode_quals_in_header,
    include_polya_in_output,
):
    def _parse_optional_int(value):
        if value is None or pd.isna(value):
            return None
        token = str(value).split(",")[0].strip()
        if token in {"", "None", "nan", "NaN"}:
            return None
        try:
            return int(float(token))
        except (TypeError, ValueError):
            return None

    result = {
        "ReadName": row["ReadName"],
        "read_length": row["read_length"],
        "cDNA_Starts": row["cDNA_Starts"],
        "cDNA_Ends": row["cDNA_Ends"],
        "cDNA_length": int(row["cDNA_Ends"]) - int(row["cDNA_Starts"]),
        "UMI_Starts": row.get("UMI_Starts"),
        "UMI_Ends": row.get("UMI_Ends"),
        "random_s_Starts": row["random_s_Starts"],
        "random_s_Ends": row["random_s_Ends"],
        "random_e_Starts": row["random_e_Starts"],
        "random_e_Ends": row["random_e_Ends"],
    }

    polyA_start = _parse_optional_int(row.get("polyA_Starts"))
    polyA_end = _parse_optional_int(row.get("polyA_Ends"))
    polyT_start = _parse_optional_int(row.get("polyT_Starts"))
    polyT_end = _parse_optional_int(row.get("polyT_Ends"))

    if polyA_start is not None and polyA_end is not None:
        result["polyA_Starts"] = row.get("polyA_Starts")
        result["polyA_Ends"] = row.get("polyA_Ends")
        result["polyA_lengths"] = polyA_end - polyA_start
    elif polyT_start is not None and polyT_end is not None:
        result["polyA_Starts"] = row.get("polyT_Starts")
        result["polyA_Ends"] = row.get("polyT_Ends")
        result["polyA_lengths"] = polyT_end - polyT_start
    else:
        result["polyA_Starts"] = None
        result["polyA_Ends"] = None
        result["polyA_lengths"] = None

    corrected_barcodes = []
    corrected_barcode_seqs = []

    for barcode_column in barcode_columns:
        whitelist = whitelist_dict[barcode_column]
        corrected_barcode, corrected_seq, min_dist, count = correct_barcode(
            row, barcode_column + "_Sequences", whitelist, threshold
        )
        result[f"corrected_{barcode_column}"] = corrected_seq
        result[f"corrected_{barcode_column}_min_dist"] = min_dist
        result[f"corrected_{barcode_column}_counts_with_min_dist"] = count
        result[f"{barcode_column}_Starts"] = row[f"{barcode_column}_Starts"]
        result[f"{barcode_column}_Ends"] = row[f"{barcode_column}_Ends"]
        corrected_barcodes.append(f"{barcode_column}:{corrected_seq}")
        corrected_barcode_seqs.append(corrected_seq)

    corrected_barcodes_str = ";".join(corrected_barcodes)

    orientation = row["orientation"]

    result["architecture"] = row["architecture"]
    result["reason"] = row["reason"]
    result["orientation"] = orientation

    cell_id, local_match_counts, local_cell_counts = assign_cell_id(result, whitelist_df, barcode_columns)
    result["cell_id"] = cell_id
    result["match_type"] = next(iter(local_match_counts.keys()), "Unknown")

    corrected_barcode_seqs_str = whitelist_dict["cell_ids"][cell_id] if cell_id != "ambiguous" else "ambiguous"

    cDNA_start = int(row["cDNA_Starts"])
    cDNA_end = int(row["cDNA_Ends"])
    cDNA_sequence = row["read"][cDNA_start:cDNA_end]
    _umi_start = _parse_optional_int(row.get("UMI_Starts"))
    _umi_end = _parse_optional_int(row.get("UMI_Ends"))
    umi_sequence = (
        row["read"][_umi_start:_umi_end]
        if (_umi_start is not None and _umi_end is not None and _umi_end > _umi_start)
        else ""
    )
    _base_q = row.get("base_qualities")
    cDNA_quality = _base_q[cDNA_start:cDNA_end] if (output_fmt == "fastq" and _base_q is not None) else None

    polya_seq = None
    polya_qual = None
    if include_polya_in_output and output_fmt in {"fastq", "fasta"}:
        polya_start_token = row.get("polyA_Starts") or row.get("polyT_Starts")
        polya_end_token = row.get("polyA_Ends") or row.get("polyT_Ends")

        try:
            polya_start = (
                int(float(str(polya_start_token).split(",")[0].strip()))
                if polya_start_token not in (None, "", "None")
                else None
            )
            polya_end = (
                int(float(str(polya_end_token).split(",")[0].strip()))
                if polya_end_token not in (None, "", "None")
                else None
            )
            if polya_start is not None and polya_end is not None and polya_end > polya_start:
                polya_seq = row["read"][polya_start:polya_end]
                if output_fmt == "fastq" and row.get("base_qualities"):
                    polya_qual = row["base_qualities"][polya_start:polya_end]
        except (ValueError, TypeError):
            polya_seq = None
            polya_qual = None

    if orientation == "+" and strand == "fwd":
        pass
    elif orientation == "-" and strand == "fwd":
        cDNA_sequence = reverse_complement(cDNA_sequence)
        umi_sequence = reverse_complement(umi_sequence)
        if polya_seq is not None:
            polya_seq = reverse_complement(polya_seq)
    elif orientation == "+" and strand == "rev":
        cDNA_sequence = reverse_complement(cDNA_sequence)
        umi_sequence = reverse_complement(umi_sequence)
        if polya_seq is not None:
            polya_seq = reverse_complement(polya_seq)
    elif orientation == "-" and strand == "rev":
        pass
    else:
        pass

    batch_reads = defaultdict(list)
    barcode_qual_suffix = ""
    if include_barcode_quals_in_header and output_fmt == "fastq":
        base_q = row.get("base_qualities", "")
        qual_tokens = []
        for barcode_column in barcode_columns:
            try:
                start_token = str(row[f"{barcode_column}_Starts"]).split(",")[0].strip()
                end_token = str(row[f"{barcode_column}_Ends"]).split(",")[0].strip()
                if start_token and end_token:
                    start = int(float(start_token))
                    end = int(float(end_token))
                    if base_q and end > start:
                        qual_slice = base_q[start:end]
                        qual_tokens.append(f"{barcode_column}:{qual_slice}")
            except (KeyError, ValueError, TypeError):
                continue

        # Add UMI qualities if present
        try:
            umi_start = int(float(str(row.get("UMI_Starts", "")).split(",")[0].strip()))
            umi_end = int(float(str(row.get("UMI_Ends", "")).split(",")[0].strip()))
            if base_q and umi_end > umi_start:
                umi_slice = base_q[umi_start:umi_end]
                if umi_slice:
                    qual_tokens.append(f"UMI:{umi_slice}")
        except (ValueError, TypeError):
            pass

        if qual_tokens:
            barcode_qual_suffix = f"|BQ:{';'.join(qual_tokens)}"

    sequence_out = cDNA_sequence
    quality_out = cDNA_quality

    if include_polya_in_output and polya_seq is not None:
        if output_fmt == "fastq" and polya_qual is None:
            # Skip appending polyA if qualities are unavailable in FASTQ mode to avoid length mismatch
            pass
        else:
            sequence_out = cDNA_sequence + polya_seq
            if output_fmt == "fastq" and polya_qual is not None:
                quality_out = (quality_out or "") + polya_qual

    _umi_name_token = f"_{umi_sequence}" if umi_sequence else ""
    _umi_field = f"|UMI:{umi_sequence}" if umi_sequence else ""

    if output_fmt == "fasta":
        demux_header = (
            f">{row['ReadName']}_{corrected_barcode_seqs_str}{_umi_name_token} "
            f"cell_id:{cell_id}|Barcodes:{corrected_barcodes_str}{_umi_field}|orientation:{orientation}"
        )
        batch_reads[corrected_barcode_seqs_str].append(
            (
                demux_header,
                sequence_out,
            )
        )
        result["demux_header"] = demux_header
        result["demux_sequence"] = sequence_out
        result["demux_quality"] = None
    elif output_fmt == "fastq":
        header = (
            f"@{row['ReadName']}_{corrected_barcode_seqs_str}{_umi_name_token} "
            f"cell_id:{cell_id}|Barcodes:{corrected_barcodes_str}{_umi_field}|orientation:{orientation}"
            f"{barcode_qual_suffix}"
        )
        _fallback_q = _base_q[cDNA_start:cDNA_end] if _base_q is not None else ""
        batch_reads[corrected_barcode_seqs_str].append(
            (
                header,
                sequence_out,
                quality_out if quality_out is not None else _fallback_q,
            )
        )
        result["demux_header"] = header
        result["demux_sequence"] = sequence_out
        result["demux_quality"] = quality_out if quality_out is not None else _fallback_q
    result["demux_bucket"] = corrected_barcode_seqs_str
    return result, local_match_counts, local_cell_counts, batch_reads


def bc_n_demultiplex(
    chunk,
    strand,
    barcode_columns,
    whitelist_dict,
    whitelist_df,
    threshold,
    output_dir,
    output_fmt,
    demuxed_fasta,
    demuxed_fasta_lock,
    ambiguous_fasta,
    ambiguous_fasta_lock,
    num_cores,
    include_barcode_quals_in_header=False,
    include_polya_in_output=False,
    write_demuxed_reads=True,
):
    args = [
        (
            row,
            strand,
            barcode_columns,
            whitelist_dict,
            whitelist_df,
            threshold,
            output_dir,
            output_fmt,
            include_barcode_quals_in_header,
            include_polya_in_output,
        )
        for _, row in chunk.iterrows()
    ]
    batch_reads = defaultdict(list)
    results = []

    if num_cores > 1:
        with Pool(num_cores) as pool:
            results = list(
                tqdm(pool.starmap(process_row, args), total=len(chunk), desc="Processing rows", disable=True)
            )

    elif num_cores == 1:
        # Loop through each row sequentially instead of using multiprocessing
        for arg in tqdm(args, total=len(chunk), desc="Processing rows (no parallelism)", disable=True):
            result = process_row(*arg)
            results.append(result)

    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()

    processed_results = [res[0] for res in results]
    all_match_type_counts = [res[1] for res in results]
    all_cell_id_counts = [res[2] for res in results]

    for res in results:
        for cell_id, reads in res[3].items():
            batch_reads[cell_id].extend(reads)

    if write_demuxed_reads:
        write_reads_to_fasta(
            batch_reads, output_fmt, demuxed_fasta, demuxed_fasta_lock, ambiguous_fasta, ambiguous_fasta_lock
        )

    match_type_counts = defaultdict(int)
    cell_id_counts = defaultdict(int)

    for match_counts in all_match_type_counts:
        for key, value in match_counts.items():
            match_type_counts[key] += value

    for cell_counts in all_cell_id_counts:
        for key, value in cell_counts.items():
            cell_id_counts[key] += value

    corrected_df = pd.DataFrame(processed_results)

    return corrected_df, match_type_counts, cell_id_counts


def generate_barcodes_stats_pdf(
    cumulative_barcodes_stats,
    barcode_columns,
    pdf_filename="barcode_plots.pdf",
    tsv_output_dir=None,
):
    plt.style.use("seaborn-v0_8-whitegrid")
    primary = "#2A6F97"
    accent = "#8AAE92"
    grid_color = "#DDE3EA"
    tsv_output_dir = tsv_output_dir or os.path.dirname(os.path.abspath(pdf_filename))
    os.makedirs(tsv_output_dir, exist_ok=True)
    all_count_rows = []
    all_min_dist_rows = []

    with PdfPages(pdf_filename) as pdf:
        for barcode_column in barcode_columns:
            count_data = pd.Series(cumulative_barcodes_stats[barcode_column]["count_data"]).sort_index()
            min_dist_data = pd.Series(cumulative_barcodes_stats[barcode_column]["min_dist_data"]).sort_index()
            count_df = pd.DataFrame(
                {
                    "barcode": barcode_column,
                    "num_matches": count_data.index.astype(str),
                    "read_count": count_data.values,
                }
            )
            min_dist_df = pd.DataFrame(
                {
                    "barcode": barcode_column,
                    "min_distance": min_dist_data.index.astype(str),
                    "read_count": min_dist_data.values,
                }
            )
            count_df.to_csv(
                os.path.join(tsv_output_dir, f"{barcode_column}_match_count_distribution.tsv"),
                sep="\t",
                index=False,
            )
            min_dist_df.to_csv(
                os.path.join(tsv_output_dir, f"{barcode_column}_min_distance_distribution.tsv"),
                sep="\t",
                index=False,
            )
            all_count_rows.append(count_df)
            all_min_dist_rows.append(min_dist_df)

            fig, axs = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")
            fig.suptitle(f"Barcode QC Summary: {barcode_column}", fontsize=14, fontweight="semibold", y=1.02)

            axs[0].bar(count_data.index, count_data.values, color=primary, edgecolor="white", linewidth=0.8)
            axs[0].set_xlabel("Number of Matches")
            axs[0].set_ylabel("Frequency")
            axs[0].set_title("Closest Whitelist Matches", fontsize=11, pad=10)
            axs[0].grid(axis="y", color=grid_color, linewidth=0.8)
            axs[0].grid(axis="x", visible=False)
            axs[0].spines["top"].set_visible(False)
            axs[0].spines["right"].set_visible(False)

            axs[1].bar(min_dist_data.index, min_dist_data.values, color=accent, edgecolor="white", linewidth=0.8)
            axs[1].set_xlabel("Minimum Distance")
            axs[1].set_ylabel("Frequency")
            axs[1].set_title("Levenshtein Distance Distribution", fontsize=11, pad=10)
            axs[1].grid(axis="y", color=grid_color, linewidth=0.8)
            axs[1].grid(axis="x", visible=False)
            axs[1].spines["top"].set_visible(False)
            axs[1].spines["right"].set_visible(False)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    pd.concat(all_count_rows, ignore_index=True).to_csv(
        os.path.join(tsv_output_dir, "barcode_match_count_distribution.tsv"),
        sep="\t",
        index=False,
    )
    pd.concat(all_min_dist_rows, ignore_index=True).to_csv(
        os.path.join(tsv_output_dir, "barcode_min_distance_distribution.tsv"),
        sep="\t",
        index=False,
    )
