import logging

logger = logging.getLogger(__name__)


def demux_wrap(input_dir, output_dir, input_file, output_fmt, strand=None, barcode_columns=None,
               include_barcode_quals=False, include_polya=False):
    """Export demultiplexed reads to FASTA/FASTQ from annotation files."""
    import os

    from scripts.export_demux import (
        _load_df,
        _gzip_file,
        _has_usable_demux_qualities,
        _has_usable_base_qualities,
        _write_from_demux_columns,
        _write_corrected_demux,
        _write_bulk_from_annotations,
    )

    if input_file is None:
        metadata_dir = f"{input_dir}/annotation_metadata"
        corrected_default = f"{metadata_dir}/annotations_valid_bc_corrected.parquet"
        annotation_default = f"{metadata_dir}/annotations_valid.parquet"
        if os.path.exists(corrected_default):
            input_file = corrected_default
        elif os.path.exists(annotation_default):
            input_file = annotation_default
        else:
            input_file = corrected_default
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Annotation file not found: {input_file}")

    df = _load_df(input_file)
    has_demux_columns = {"demux_bucket", "demux_header", "demux_sequence", "cell_id"}.issubset(set(df.columns))
    has_bulk_columns = {"ReadName", "read", "cDNA_Starts", "cDNA_Ends"}.issubset(set(df.columns))
    if not has_demux_columns and not has_bulk_columns:
        raise ValueError(
            "Input file does not contain demux columns or bulk annotation columns required for FASTA/FASTQ export."
        )

    os.makedirs(output_dir, exist_ok=True)
    out_dir = f"{output_dir}/demuxed_fasta"
    os.makedirs(out_dir, exist_ok=True)

    effective_output_fmt = output_fmt
    if output_fmt == "fastq":
        has_quality = _has_usable_demux_qualities(df) if has_demux_columns else _has_usable_base_qualities(df)
        if not has_quality:
            logger.warning("Base quality scores not available; requested FASTQ demux output will be written as FASTA.")
            effective_output_fmt = "fasta"

    if effective_output_fmt == "fastq":
        demuxed_path = f"{out_dir}/demuxed.fastq"
        ambiguous_path = f"{out_dir}/ambiguous.fastq"
    else:
        demuxed_path = f"{out_dir}/demuxed.fasta"
        ambiguous_path = f"{out_dir}/ambiguous.fasta"

    has_corrected_columns = "cell_id" in df.columns and any(c.startswith("corrected_") for c in df.columns)

    if has_demux_columns:
        _write_from_demux_columns(df, effective_output_fmt, demuxed_path, ambiguous_path)
    elif has_corrected_columns and strand and barcode_columns:
        logger.info("Writing demux from corrected annotations with full headers.")
        _write_corrected_demux(
            df.iter_rows(named=True), effective_output_fmt, demuxed_path, ambiguous_path,
            strand, barcode_columns, include_barcode_quals, include_polya,
        )
    else:
        logger.info("Demux columns not found; exporting bulk reads from annotation coordinates.")
        _write_bulk_from_annotations(df, effective_output_fmt, demuxed_path, ambiguous_path)

    _gzip_file(demuxed_path)
    _gzip_file(ambiguous_path)
