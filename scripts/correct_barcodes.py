import logging
import gc
import gzip
import os
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from rapidfuzz import process
from multiprocessing import Pool
from collections import defaultdict
from rapidfuzz.distance import Levenshtein

from scripts.demultiplex import assign_cell_id

logger = logging.getLogger(__name__)


def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(complement.get(base, base) for base in reversed(seq))


def correct_barcode(row, column_name, whitelist, threshold):
    """Find the closest whitelist barcode(s) using Levenshtein distance."""
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
    """Write batched reads to gzipped demuxed/ambiguous FASTA or FASTQ files."""
    for cell_id, reads in batch_reads.items():
        if cell_id == "ambiguous":
            if ambiguous_fasta_lock:
                with ambiguous_fasta_lock:
                    fasta_file = gzip.open(ambiguous_fasta, "at")
            else:
                fasta_file = gzip.open(ambiguous_fasta, "at")
        else:
            if demuxed_fasta_lock:
                with demuxed_fasta_lock:
                    fasta_file = gzip.open(demuxed_fasta, "at")
            else:
                fasta_file = gzip.open(demuxed_fasta, "at")

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
    """Correct barcodes, assign cell ID, and build demux output for a single annotation row."""
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

    cDNA_start = _parse_optional_int(row.get("cDNA_Starts"))
    cDNA_end = _parse_optional_int(row.get("cDNA_Ends"))
    if cDNA_start is None or cDNA_end is None or cDNA_end <= cDNA_start:
        raise ValueError(
            f"Invalid cDNA coordinates for read {row.get('ReadName', 'unknown')}: "
            f"cDNA_Starts={row.get('cDNA_Starts')}, cDNA_Ends={row.get('cDNA_Ends')}"
        )

    # Start from the original row so corrected outputs retain all annotation columns
    # (including barcode *_Sequences) and then append correction/demux fields.
    result = row.to_dict()
    result["cDNA_length"] = cDNA_end - cDNA_start

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
        result[f"{barcode_column}_Sequences"] = row.get(f"{barcode_column}_Sequences")
        result[f"{barcode_column}_Starts"] = row[f"{barcode_column}_Starts"]
        result[f"{barcode_column}_Ends"] = row[f"{barcode_column}_Ends"]
        corrected_barcodes.append(f"{barcode_column}:{corrected_seq}")
        corrected_barcode_seqs.append(corrected_seq)

    corrected_barcodes_str = ";".join(corrected_barcodes)

    orientation = row["orientation"]

    result["architecture"] = row["architecture"]
    result["reason"] = row["reason"]
    result["orientation"] = orientation

    cell_id = assign_cell_id(result, whitelist_df, barcode_columns)
    result["cell_id"] = cell_id
    result["match_type"] = "Exact match" if cell_id != "ambiguous" else "Ambiguous"

    corrected_barcode_seqs_str = whitelist_dict["cell_ids"][cell_id] if cell_id != "ambiguous" else "ambiguous"

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
    return result, batch_reads


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
    """Correct barcodes and demultiplex a chunk of annotated reads in parallel."""
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

    for res in results:
        for cell_id, reads in res[1].items():
            batch_reads[cell_id].extend(reads)

    if write_demuxed_reads:
        write_reads_to_fasta(
            batch_reads, output_fmt, demuxed_fasta, demuxed_fasta_lock, ambiguous_fasta, ambiguous_fasta_lock
        )

    return pd.DataFrame(processed_results)
