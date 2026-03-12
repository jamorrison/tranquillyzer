import os
import gc
import csv
import gzip
import shutil
import logging
import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf
from scripts.correct_barcodes import bc_n_demultiplex
from scripts.extract_annotated_seqs import extract_annotated_full_length_seqs

logger = logging.getLogger(__name__)


def _gzip_chunk_output(path):
    if not path or not os.path.exists(path):
        return None
    gz_path = path + ".gz"
    if os.path.exists(gz_path):
        os.remove(gz_path)
    with open(path, "rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(path)
    return gz_path


def _parse_first_int(value):
    if value is None or pd.isna(value):
        return None
    token = str(value).split(",")[0].strip()
    if token in {"", "None", "nan", "NaN"}:
        return None
    try:
        return int(float(token))
    except (TypeError, ValueError):
        return None


def _bulk_export_record(row, output_fmt, include_polya=False):
    cDNA_start = _parse_first_int(row.get("cDNA_Starts"))
    cDNA_end = _parse_first_int(row.get("cDNA_Ends"))
    read = row.get("read")
    if cDNA_start is None or cDNA_end is None or read is None or cDNA_end <= cDNA_start:
        return None

    sequence = str(read)[cDNA_start:cDNA_end]
    quality = None
    if output_fmt == "fastq":
        base_q = row.get("base_qualities")
        quality = str(base_q)[cDNA_start:cDNA_end] if base_q is not None and not pd.isna(base_q) else ""

    if include_polya:
        polya_start = _parse_first_int(row.get("polyA_Starts"))
        polya_end = _parse_first_int(row.get("polyA_Ends"))
        if polya_start is None or polya_end is None:
            polya_start = _parse_first_int(row.get("polyT_Starts"))
            polya_end = _parse_first_int(row.get("polyT_Ends"))
        if (
            polya_start is not None
            and polya_end is not None
            and polya_end > polya_start
            and polya_end <= len(str(read))
        ):
            polya_seq = str(read)[polya_start:polya_end]
            sequence = sequence + polya_seq
            if output_fmt == "fastq":
                base_q = row.get("base_qualities")
                polya_q = str(base_q)[polya_start:polya_end] if base_q is not None and not pd.isna(base_q) else ""
                quality = (quality or "") + polya_q

    orientation = row.get("orientation", "NA")
    read_name = row.get("ReadName", "read")

    if output_fmt == "fastq":
        header = f"@{read_name} orientation:{orientation}"
        quality = quality or ""
        if len(quality) < len(sequence):
            quality = quality + ("!" * (len(sequence) - len(quality)))
        elif len(quality) > len(sequence):
            quality = quality[: len(sequence)]
        return header, sequence, quality

    header = f">{read_name} orientation:{orientation}"
    return header, sequence


def _chunk_demux_paths(chunk_output_dir, pass_num, bin_name, chunk_idx, output_fmt):
    if chunk_output_dir is None:
        return None, None
    ext = "fastq" if output_fmt == "fastq" else "fasta"
    demux_dir = os.path.join(chunk_output_dir, "demuxed_chunks")
    ambiguous_dir = os.path.join(chunk_output_dir, "ambiguous_chunks")
    os.makedirs(demux_dir, exist_ok=True)
    os.makedirs(ambiguous_dir, exist_ok=True)
    chunk_stub = f"pass{pass_num}__{bin_name}__chunk{int(chunk_idx):06d}.{ext}"
    return os.path.join(demux_dir, chunk_stub), os.path.join(ambiguous_dir, chunk_stub)


def process_full_length_reads_in_chunks_and_save(
    reads,
    original_read_names,
    strand,
    output_fmt,
    base_qualities,
    model_type,
    pass_num,
    model_path_w_CRF,
    predictions,
    bin_name,
    chunk_idx,
    label_binarizer,
    cumulative_barcodes_stats,
    actual_lengths,
    seq_order,
    add_header,
    output_dir,
    invalid_output_file,
    invalid_file_lock,
    valid_output_file,
    valid_file_lock,
    barcodes,
    whitelist_df,
    whitelist_dict,
    demuxed_fasta,
    demuxed_fasta_lock,
    ambiguous_fasta,
    ambiguous_fasta_lock,
    threshold,
    n_jobs,
    include_barcode_quals,
    include_polya,
    run_barcode_correction=True,
    run_demux=True,
    chunk_output_dir=None,
):
    reads_in_chunk = len(reads)

    logger.info(f"Post-processing {bin_name} chunk - {chunk_idx}: number of reads = {reads_in_chunk}")

    n_jobs_extract = min(16, reads_in_chunk)
    chunk_contiguous_annotated_sequences = extract_annotated_full_length_seqs(
        reads, predictions, model_path_w_CRF, actual_lengths, label_binarizer, seq_order, barcodes, n_jobs_extract
    )

    chunk_df = pd.DataFrame.from_records(
        (
            {
                "ReadName": original_read_names[i],
                "read_length": annotated_read["read_length"],
                "read": annotated_read["read"],
                **{
                    f"{label}_Starts": ", ".join(map(str, annotations["Starts"]))
                    for label, annotations in annotated_read.items()
                    if label not in {"architecture", "reason"} and "Starts" in annotations
                },
                **{
                    f"{label}_Ends": ", ".join(map(str, annotations["Ends"]))
                    for label, annotations in annotated_read.items()
                    if label not in {"architecture", "reason"} and "Ends" in annotations
                },
                **{
                    f"{label}_Sequences": ", ".join(map(str, annotated_read[label]["Sequences"]))
                    for label in barcodes
                    if label in annotated_read and "Sequences" in annotated_read[label]
                },
                "base_qualities": base_qualities[i] if output_fmt == "fastq" else None,
                "architecture": annotated_read["architecture"],
                "reason": annotated_read["reason"],
                "orientation": annotated_read["orientation"],
            }
            for i, annotated_read in enumerate(chunk_contiguous_annotated_sequences)
        )
    )
    # Filter out invalid reads
    invalid_reads_df = chunk_df[chunk_df["architecture"] == "invalid"]
    valid_reads_df = chunk_df[chunk_df["architecture"] != "invalid"]

    if run_demux:
        demuxed_chunk_file, ambiguous_chunk_file = _chunk_demux_paths(
            chunk_output_dir, pass_num, bin_name, chunk_idx, output_fmt
        )
    else:
        demuxed_chunk_file, ambiguous_chunk_file = None, None

    if model_type == "HYB" and pass_num == 1:
        tmp_invalid_dir = os.path.join(output_dir, "tmp_invalid_reads")
        os.makedirs(tmp_invalid_dir, exist_ok=True)

        tmp_invalid_payload = {
            "ReadName": invalid_reads_df["ReadName"],
            "read": invalid_reads_df["read"],
            "read_length": invalid_reads_df["read_length"],
        }
        if output_fmt == "fastq":
            tmp_invalid_payload["base_qualities"] = invalid_reads_df["base_qualities"]

        tmp_invalid_df = pl.DataFrame(tmp_invalid_payload)

        tmp_path = f"{tmp_invalid_dir}/{bin_name}.tsv"
        lock_path = f"{tmp_path}.lock"

        if not os.path.exists(lock_path):
            with open(lock_path, "w") as lock_file:
                lock_file.write("")

        write_header = not os.path.exists(tmp_path)
        with open(tmp_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            if write_header:
                writer.writerow(tmp_invalid_df.columns)
            writer.writerows(tmp_invalid_df.rows())

    else:
        if not invalid_reads_df.empty:
            os.makedirs(os.path.join(chunk_output_dir, "invalid_chunks"), exist_ok=True)
            invalid_chunk_file = os.path.join(
                chunk_output_dir, "invalid_chunks", f"pass{pass_num}__{bin_name}__chunk{int(chunk_idx):06d}.tsv"
            )
            invalid_reads_df.to_csv(invalid_chunk_file, sep="\t", index=False)

    column_mapping = {barcode: barcode for barcode in barcodes}

    if not valid_reads_df.empty:
        if run_barcode_correction:
            corrected_df, _, _ = bc_n_demultiplex(
                valid_reads_df,
                strand,
                list(column_mapping.keys()),
                whitelist_dict,
                whitelist_df,
                threshold,
                output_dir,
                output_fmt,
                demuxed_chunk_file,
                None,
                ambiguous_chunk_file,
                None,
                n_jobs,
                include_barcode_quals,
                include_polya,
                write_demuxed_reads=run_demux,
            )
            output_df = corrected_df
        else:
            output_df = valid_reads_df.copy()
            output_df["cDNA_length"] = output_df.apply(
                lambda row: int(float(str(row["cDNA_Ends"]).split(",")[0].strip()))
                - int(float(str(row["cDNA_Starts"]).split(",")[0].strip())),
                axis=1,
            )
            if run_demux and demuxed_chunk_file:
                bulk_reads = []
                for _, row in valid_reads_df.iterrows():
                    record = _bulk_export_record(row, output_fmt, include_polya)
                    if record is not None:
                        bulk_reads.append(record)
                if bulk_reads:
                    with open(demuxed_chunk_file, "a") as out_fh:
                        if output_fmt == "fastq":
                            for header, sequence, quality in bulk_reads:
                                out_fh.write(f"{header}\n{sequence}\n+\n{quality}\n")
                        else:
                            for header, sequence in bulk_reads:
                                out_fh.write(f"{header}\n{sequence}\n")

        os.makedirs(os.path.join(chunk_output_dir, "valid_chunks"), exist_ok=True)
        valid_chunk_file = os.path.join(
            chunk_output_dir, "valid_chunks", f"pass{pass_num}__{bin_name}__chunk{int(chunk_idx):06d}.tsv"
        )
        output_df.to_csv(valid_chunk_file, sep="\t", index=False)

        logger.info(f"Post-processed {bin_name} chunk - {chunk_idx}: number of reads = {reads_in_chunk}")

    if run_demux:
        _gzip_chunk_output(demuxed_chunk_file)
        _gzip_chunk_output(ambiguous_chunk_file)

    for local_df in ["chunk_df", "corrected_df", "invalid_reads_df", "valid_reads_df"]:
        if local_df:
            del local_df

    os.makedirs(os.path.join(chunk_output_dir, "done"), exist_ok=True)
    done_file = os.path.join(chunk_output_dir, "done", f"pass{pass_num}__{bin_name}__chunk{int(chunk_idx):06d}.done")
    with open(done_file, "w") as f:
        f.write("done\n")

    # Clean up after each chunk to free memory
    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()


def post_process_reads(
    reads,
    read_names,
    strand,
    output_fmt,
    base_qualities,
    model_type,
    pass_num,
    model_path_w_CRF,
    predictions,
    label_binarizer,
    cumulative_barcodes_stats,
    read_lengths,
    seq_order,
    add_header,
    bin_name,
    chunk_idx,
    output_dir,
    invalid_output_file,
    invalid_file_lock,
    valid_output_file,
    valid_file_lock,
    barcodes,
    whitelist_df,
    whitelist_dict,
    threshold,
    checkpoint_file,
    chunk_start,
    match_type_counter,
    cell_id_counter,
    demuxed_fasta,
    demuxed_fasta_lock,
    ambiguous_fasta,
    ambiguous_fasta_lock,
    njobs,
    include_barcode_quals,
    include_polya,
    run_barcode_correction=True,
    run_demux=True,
    chunk_output_dir=None,
):
    process_full_length_reads_in_chunks_and_save(
        reads,
        read_names,
        strand,
        output_fmt,
        base_qualities,
        model_type,
        pass_num,
        model_path_w_CRF,
        predictions,
        bin_name,
        chunk_idx,
        label_binarizer,
        cumulative_barcodes_stats,
        read_lengths,
        seq_order,
        add_header,
        output_dir,
        invalid_output_file,
        invalid_file_lock,
        valid_output_file,
        valid_file_lock,
        barcodes,
        whitelist_df,
        whitelist_dict,
        demuxed_fasta,
        demuxed_fasta_lock,
        ambiguous_fasta,
        ambiguous_fasta_lock,
        threshold,
        njobs,
        include_barcode_quals,
        include_polya,
        run_barcode_correction,
        run_demux,
        chunk_output_dir,
    )

    gc.collect()  # Clean up memory after processing each chunk

    return True


def filtering_reason_stats(reason_counter_by_bin, output_dir):
    # Convert dictionary to DataFrame (Bins as Columns, Reasons as Rows)
    raw_counts_df = pd.DataFrame.from_dict(reason_counter_by_bin, orient="index").fillna(0).T

    # Compute total reads per bin
    total_reads = raw_counts_df.sum(axis=0)

    # Normalize each column (fraction per bin)
    normalized_data = raw_counts_df.div(total_reads, axis=1)

    # Save both raw counts and normalized fractions
    raw_counts_df.to_csv(f"{output_dir}/filtered_raw_counts_by_bins.tsv", sep="\t")
    normalized_data.to_csv(f"{output_dir}/filtered_normalized_fractions_by_bins.tsv", sep="\t")

    print(f"Saved raw counts to {output_dir}/filtered_raw_counts_by_bins.tsv")
    print(f"Saved normalized fractions to {output_dir}/filtered_normalized_fractions_by_bins.tsv")


def plot_read_n_cDNA_lengths(output_dir):

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    df = pl.read_parquet(f"{output_dir}/annotations_valid.parquet", columns=["read_length", "cDNA_length"])
    read_lengths = []
    cDNA_lengths = []

    read_lengths.extend(df["read_length"].to_list())
    read_lengths = np.array(read_lengths, dtype=int)

    cDNA_lengths.extend(df["cDNA_length"].to_list())
    cDNA_lengths = np.array(cDNA_lengths, dtype=int)

    log_read_lengths = np.log10(read_lengths[read_lengths > 0])
    log_cDNA_lengths = np.log10(cDNA_lengths[cDNA_lengths > 0])

    with PdfPages(f"{output_dir}/plots/cDNA_len_distr.pdf") as pdf:
        # valid read length distribution
        if len(log_read_lengths[log_read_lengths > 0]):
            plt.figure(figsize=(8, 6))
            plt.hist(log_read_lengths[log_read_lengths > 0], bins=100, color="blue", edgecolor="black")
            plt.title("Read Length Distribution (Log Scale)")
            plt.xlabel("Log10(Read Length)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # cDNA length distribution
        if len(log_cDNA_lengths[log_cDNA_lengths > 0]):
            plt.figure(figsize=(8, 6))
            plt.hist(log_cDNA_lengths[log_cDNA_lengths > 0], bins=100, color="blue", edgecolor="black")
            plt.title("cDNA Length Distribution (Log Scale)")
            plt.xlabel("Log10(cDNA Length)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
