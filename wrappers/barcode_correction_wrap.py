import logging

logger = logging.getLogger(__name__)


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
    """Orchestrate barcode correction on annotated reads with optional demux."""
    from scripts.barcode_correction import (
        load_libs,
        _scan_annotations_in_chunks,
        _combine_demux_chunk_outputs,
        _has_usable_base_qualities,
        _infer_barcode_columns,
    )

    (
        os,
        shutil,
        pd,
        pl,
        chain,
        bc_n_demultiplex,
        seq_orders,
    ) = load_libs()

    os.makedirs(output_dir, exist_ok=True)

    if input_file is None:
        annotation_default = f"{input_dir}/annotations_valid.parquet"
        corrected_default = f"{input_dir}/annotations_valid_bc_corrected.parquet"
        if os.path.exists(annotation_default):
            input_file = annotation_default
        elif os.path.exists(corrected_default):
            input_file = corrected_default
        else:
            input_file = annotation_default
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

    barcode_columns, strand = _infer_barcode_columns(first_chunk.columns, whitelist_df, seq_order_file, model_name, seq_orders)
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
    in_place_corrected_overwrite = os.path.abspath(input_file) == os.path.abspath(corrected_parquet)
    corrected_parquet_tmp = (
        f"{corrected_parquet}.tmp" if in_place_corrected_overwrite else corrected_parquet
    )
    if os.path.exists(corrected_tsv):
        os.remove(corrected_tsv)
    if os.path.exists(corrected_parquet_tmp):
        os.remove(corrected_parquet_tmp)
    if os.path.exists(corrected_parquet) and not in_place_corrected_overwrite:
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
            demuxed_fasta = os.path.join(fasta_dir, "demuxed.fastq.gz")
            ambiguous_fasta = os.path.join(fasta_dir, "ambiguous.fastq.gz")
            demux_ext = "fastq"
        else:
            demuxed_fasta = os.path.join(fasta_dir, "demuxed.fasta.gz")
            ambiguous_fasta = os.path.join(fasta_dir, "ambiguous.fasta.gz")
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
            chunk_name = f"chunk{chunk_idx:06d}.{demux_ext}.gz"
            chunk_demuxed = os.path.join(demux_chunk_dir, chunk_name)
            chunk_ambiguous = os.path.join(ambiguous_chunk_dir, chunk_name)
        corrected_df = bc_n_demultiplex(
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

    pl.scan_csv(corrected_tsv, separator="\t", infer_schema_length=5000).sink_parquet(
        corrected_parquet_tmp, compression="snappy"
    )
    if in_place_corrected_overwrite:
        os.replace(corrected_parquet_tmp, corrected_parquet)
    if os.path.exists(corrected_tsv):
        os.remove(corrected_tsv)
    if os.path.basename(input_file) == "annotations_valid.parquet" and os.path.exists(input_file):
        os.remove(input_file)

    logger.info(f"Wrote corrected annotations to {corrected_parquet}")
