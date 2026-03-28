import logging

logger = logging.getLogger(__name__)


def load_libs():
    """Lazily import and return libraries needed by barcode correction."""
    import os
    import shutil
    from itertools import chain

    import pandas as pd
    import polars as pl

    from scripts.correct_barcodes import bc_n_demultiplex
    from scripts.trained_models import seq_orders

    return (
        os,
        shutil,
        pd,
        pl,
        chain,
        bc_n_demultiplex,
        seq_orders,
    )


def _scan_annotations_in_chunks(input_file, chunk_size):
    """Yield annotation DataFrame chunks of the given size from a Parquet or TSV file."""
    import polars as pl

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
    """Concatenate per-chunk demux gzip files into final demuxed/ambiguous outputs."""
    import os
    import shutil

    def _sorted_files(path):
        if not os.path.isdir(path):
            return []
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(f".{ext}.gz")]
        files.sort()
        return files

    def _concat(src_files, out_path):
        if not src_files:
            return
        with open(out_path, "wb") as out_fh:
            for src in src_files:
                with open(src, "rb") as src_fh:
                    shutil.copyfileobj(src_fh, out_fh)

    demux_files = _sorted_files(demux_chunk_dir)
    amb_files = _sorted_files(ambiguous_chunk_dir)
    _concat(demux_files, demuxed_fasta)
    _concat(amb_files, ambiguous_fasta)
    if not keep_demux_chunk_outputs_after_combine:
        for path in demux_files + amb_files:
            os.remove(path)


def _has_usable_base_qualities(lazy_df):
    """Return True if the lazy DataFrame has at least one non-null base_qualities value."""
    import polars as pl

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


def _infer_barcode_columns(columns, whitelist_df, seq_order_file, model_name, seq_orders):
    """Infer barcode column names from seq_orders config or annotation/whitelist column overlap."""
    if seq_order_file:
        _, _, barcodes, _, strand = seq_orders(seq_order_file, model_name)
        if barcodes:
            return barcodes, strand

    barcode_seq_cols = {col.replace("_Sequences", "") for col in columns if col.endswith("_Sequences")}
    whitelist_cols = set(whitelist_df.columns.tolist())
    inferred = [col for col in whitelist_df.columns if col in barcode_seq_cols and col in whitelist_cols]
    return inferred, "fwd"
