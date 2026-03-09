import logging
import os
from itertools import chain
import gzip
import shutil

import pandas as pd
import polars as pl

from scripts.correct_barcodes import bc_n_demultiplex
from scripts.trained_models import seq_orders

logger = logging.getLogger(__name__)


def _gzip_file(path):
    if path is None or not os.path.exists(path):
        return
    gz_path = path + ".gz"
    if os.path.exists(gz_path):
        os.remove(gz_path)
    with open(path, "rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(path)


def _scan_annotations_in_chunks(input_file, chunk_size):
    if input_file.endswith(".parquet"):
        lazy_df = pl.scan_parquet(input_file)
    else:
        lazy_df = pl.scan_csv(input_file, separator="\t", infer_schema_length=5000)

    offset = 0
    while True:
        chunk = lazy_df.slice(offset, chunk_size).collect()
        if chunk.height == 0:
            break
        yield chunk
        offset += chunk_size


def _combine_demux_chunk_outputs(
    demux_chunk_dir,
    ambiguous_chunk_dir,
    demuxed_fasta,
    ambiguous_fasta,
    ext,
    keep_demux_chunk_outputs_after_combine,
):
    def _sorted_files(path):
        if not os.path.isdir(path):
            return []
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(f".{ext}")]
        files.sort()
        return files

    def _concat(src_files, out_path):
        if not src_files:
            return
        with open(out_path, "w") as out_fh:
            for src in src_files:
                with open(src, "r") as src_fh:
                    out_fh.write(src_fh.read())

    demux_files = _sorted_files(demux_chunk_dir)
    amb_files = _sorted_files(ambiguous_chunk_dir)
    _concat(demux_files, demuxed_fasta)
    _concat(amb_files, ambiguous_fasta)
    if not keep_demux_chunk_outputs_after_combine:
        for path in demux_files + amb_files:
            os.remove(path)


def _has_usable_base_qualities(lazy_df):
    schema = lazy_df.collect_schema().names()
    if "base_qualities" not in schema:
        return False
    probe = (
        lazy_df.select(
            (
                pl.col("base_qualities").is_not_null() & (pl.col("base_qualities").cast(pl.Utf8).str.len_chars() > 0)
            )
            .any()
            .alias("has_bq")
        )
        .collect()
        .item()
    )
    return bool(probe)


def _infer_barcode_columns(columns, whitelist_df, seq_order_file, model_name):
    if seq_order_file:
        _, _, barcodes, _, strand = seq_orders(seq_order_file, model_name)
        if barcodes:
            return barcodes, strand

    barcode_seq_cols = {col.replace("_Sequences", "") for col in columns if col.endswith("_Sequences")}
    whitelist_cols = set(whitelist_df.columns.tolist())
    inferred = [col for col in whitelist_df.columns if col in barcode_seq_cols and col in whitelist_cols]
    return inferred, "fwd"


def barcode_correction_wrap(
    input_dir,
    whitelist_file,
    output_dir,
    input_file,
    output_fmt,
    seq_order_file,
    model_name,
    bc_lv_threshold,
    threads,
    chunk_size,
    include_barcode_quals,
    include_polya,
    run_demux,
    keep_demux_chunk_outputs_after_combine=False,
):
    os.makedirs(output_dir, exist_ok=True)

    input_file = input_file or f"{input_dir}/annotations_valid.parquet"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Annotation file not found: {input_file}")

    whitelist_df = pd.read_csv(whitelist_file, sep="\t")
    if input_file.endswith(".parquet"):
        lazy_df = pl.scan_parquet(input_file)
    else:
        lazy_df = pl.scan_csv(input_file, separator="\t", infer_schema_length=5000)

    effective_output_fmt = output_fmt
    if run_demux and output_fmt == "fastq":
        if not _has_usable_base_qualities(lazy_df):
            logger.warning("Base quality scores not available; requested FASTQ demux output will be written as FASTA.")
            effective_output_fmt = "fasta"

    chunk_iter = _scan_annotations_in_chunks(input_file, chunk_size)
    first_chunk = next(chunk_iter, None)
    if first_chunk is None or first_chunk.height == 0:
        raise ValueError(f"No rows found in annotation file: {input_file}")

    barcode_columns, strand = _infer_barcode_columns(first_chunk.columns, whitelist_df, seq_order_file, model_name)
    if not barcode_columns:
        raise ValueError(
            "Could not infer barcode columns. Provide --seq-order-file/--model-name or ensure whitelist columns "
            "match annotation '*_Sequences' columns."
        )

    whitelist_dict = {
        "cell_ids": {
            idx + 1: "-".join(map(str, row.dropna().unique())) for idx, row in whitelist_df[barcode_columns].iterrows()
        },
        **{barcode: whitelist_df[barcode].dropna().unique().tolist() for barcode in barcode_columns},
    }

    corrected_tsv = f"{output_dir}/annotations_valid_bc_corrected.tsv"
    corrected_parquet = f"{output_dir}/annotations_valid_bc_corrected.parquet"
    if os.path.exists(corrected_tsv):
        os.remove(corrected_tsv)
    if os.path.exists(corrected_parquet):
        os.remove(corrected_parquet)

    demuxed_fasta = None
    ambiguous_fasta = None
    demux_chunk_dir = None
    ambiguous_chunk_dir = None
    demux_ext = None
    if run_demux:
        fasta_dir = os.path.join(output_dir, "demuxed_fasta")
        os.makedirs(fasta_dir, exist_ok=True)
        if effective_output_fmt == "fastq":
            demuxed_fasta = os.path.join(fasta_dir, "demuxed.fastq")
            ambiguous_fasta = os.path.join(fasta_dir, "ambiguous.fastq")
            demux_ext = "fastq"
        else:
            demuxed_fasta = os.path.join(fasta_dir, "demuxed.fasta")
            ambiguous_fasta = os.path.join(fasta_dir, "ambiguous.fasta")
            demux_ext = "fasta"
        demux_chunk_dir = os.path.join(fasta_dir, "demuxed_chunks")
        ambiguous_chunk_dir = os.path.join(fasta_dir, "ambiguous_chunks")
        os.makedirs(demux_chunk_dir, exist_ok=True)
        os.makedirs(ambiguous_chunk_dir, exist_ok=True)

    first_write = True
    chunk_idx = 0

    for chunk_df in chain([first_chunk], chunk_iter):
        chunk_idx += 1
        chunk_pd = pd.DataFrame(chunk_df.to_dicts())
        chunk_demuxed = None
        chunk_ambiguous = None
        if run_demux:
            chunk_name = f"chunk{chunk_idx:06d}.{demux_ext}"
            chunk_demuxed = os.path.join(demux_chunk_dir, chunk_name)
            chunk_ambiguous = os.path.join(ambiguous_chunk_dir, chunk_name)
        corrected_df, _, _ = bc_n_demultiplex(
            chunk_pd,
            strand,
            barcode_columns,
            whitelist_dict,
            whitelist_df,
            bc_lv_threshold,
            output_dir,
            effective_output_fmt,
            chunk_demuxed,
            None,
            chunk_ambiguous,
            None,
            threads,
            include_barcode_quals_in_header=include_barcode_quals,
            include_polya_in_output=include_polya,
            write_demuxed_reads=run_demux,
        )

        corrected_df.to_csv(corrected_tsv, sep="\t", index=False, mode="w" if first_write else "a", header=first_write)
        first_write = False

    if run_demux:
        _combine_demux_chunk_outputs(
            demux_chunk_dir,
            ambiguous_chunk_dir,
            demuxed_fasta,
            ambiguous_fasta,
            demux_ext,
            keep_demux_chunk_outputs_after_combine,
        )
        _gzip_file(demuxed_fasta)
        _gzip_file(ambiguous_fasta)

    pl.scan_csv(corrected_tsv, separator="\t", infer_schema_length=5000).sink_parquet(
        corrected_parquet, compression="snappy"
    )
    if os.path.exists(corrected_tsv):
        os.remove(corrected_tsv)
    if os.path.basename(input_file) == "annotations_valid.parquet" and os.path.exists(input_file):
        os.remove(input_file)

    logger.info(f"Wrote corrected annotations to {corrected_parquet}")
