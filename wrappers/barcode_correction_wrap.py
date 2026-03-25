import logging
import os
import queue
import shutil
import threading

import pandas as pd

logger = logging.getLogger(__name__)


def _parse_first_int(value):
    """Extract the first integer from a comma-separated string, or return None."""
    if value is None:
        return None
    token = str(value).split(",")[0].strip()
    if token in {"", "None", "nan", "NaN"}:
        return None
    try:
        return int(float(token))
    except (TypeError, ValueError):
        return None


def _correct_and_demux_row(row_data, barcode_columns, whitelist_dict, whitelist_sets,
                            cell_id_lookup, whitelist_df, threshold, strand, output_fmt,
                            include_barcode_quals, include_polya, run_demux):
    """Pool worker: correct barcodes + build demux record for one row.

    Returns (correction_dict, demux_record_str_or_None, is_ambiguous).
    """
    from scripts.correct_barcodes import correct_barcode, reverse_complement

    (read_name, read_seq, orientation, bc_seqs,
     cs, ce, us, ue, base_q,
     pa_s, pa_e, bc_starts_ends) = row_data

    # --- 1. Barcode correction: exact-match short-circuit + fuzzy ---
    correction = {}
    corrected_parts = []
    corrected_seqs = []
    for bc_col in barcode_columns:
        seq = bc_seqs.get(bc_col)
        if not seq or str(seq) in ("", "nan", "None", "NaN"):
            correction[f"corrected_{bc_col}"] = "NMF"
            correction[f"corrected_{bc_col}_min_dist"] = -1
            correction[f"corrected_{bc_col}_counts_with_min_dist"] = 0
            corrected_parts.append(f"{bc_col}:NMF")
            corrected_seqs.append("NMF")
            continue

        wl_set = whitelist_sets[bc_col]
        if seq in wl_set:
            correction[f"corrected_{bc_col}"] = seq
            correction[f"corrected_{bc_col}_min_dist"] = 0
            correction[f"corrected_{bc_col}_counts_with_min_dist"] = 1
            corrected_parts.append(f"{bc_col}:{seq}")
            corrected_seqs.append(seq)
            continue
        rc_seq = reverse_complement(seq)
        if rc_seq in wl_set:
            correction[f"corrected_{bc_col}"] = rc_seq
            correction[f"corrected_{bc_col}_min_dist"] = 0
            correction[f"corrected_{bc_col}_counts_with_min_dist"] = 1
            corrected_parts.append(f"{bc_col}:{rc_seq}")
            corrected_seqs.append(rc_seq)
            continue

        _, corrected, min_dist, count = correct_barcode(
            {f"{bc_col}_Sequences": seq}, f"{bc_col}_Sequences",
            whitelist_dict[bc_col], threshold,
        )
        correction[f"corrected_{bc_col}"] = corrected
        correction[f"corrected_{bc_col}_min_dist"] = min_dist
        correction[f"corrected_{bc_col}_counts_with_min_dist"] = count
        corrected_parts.append(f"{bc_col}:{corrected}")
        corrected_seqs.append(corrected)

    # --- 2. Assign cell_id ---
    if cell_id_lookup is not None and len(barcode_columns) == 1:
        # Fast path: single-barcode dict lookup
        corr = corrected_seqs[0]
        cell_id = cell_id_lookup.get(corr, "ambiguous")
    else:
        # Multi-barcode: product match via assign_cell_id
        from scripts.demultiplex import assign_cell_id
        cell_id = assign_cell_id(correction, whitelist_df, barcode_columns)
    correction["cell_id"] = str(cell_id)
    correction["match_type"] = "Exact match" if cell_id != "ambiguous" else "Ambiguous"
    is_ambiguous = cell_id == "ambiguous"

    # --- 3. Build demux record (if requested) ---
    demux_record = None
    if run_demux and cs is not None and ce is not None and read_seq and ce > cs:
        cDNA_seq = read_seq[cs:ce]
        umi_seq = read_seq[us:ue] if (us is not None and ue is not None and ue > us) else ""

        needs_rc = (orientation == "-" and strand == "fwd") or (orientation == "+" and strand == "rev")
        if needs_rc:
            cDNA_seq = reverse_complement(cDNA_seq)
            if umi_seq:
                umi_seq = reverse_complement(umi_seq)

        bc_str = ";".join(corrected_parts)
        cell_id_str = str(cell_id) if cell_id != "ambiguous" else "ambiguous"
        umi_name = f"_{umi_seq}" if umi_seq else ""
        umi_field = f"|UMI:{umi_seq}" if umi_seq else ""

        # PolyA
        seq_out = cDNA_seq
        if include_polya and pa_s is not None and pa_e is not None and pa_e > pa_s:
            polya_seq = read_seq[pa_s:pa_e]
            if needs_rc:
                polya_seq = reverse_complement(polya_seq)
            seq_out = cDNA_seq + polya_seq

        if output_fmt == "fastq":
            qual = str(base_q)[cs:ce] if base_q is not None else ""
            if include_polya and pa_s is not None and pa_e is not None and pa_e > pa_s:
                polya_qual = str(base_q)[pa_s:pa_e] if base_q is not None else ""
                qual = qual + polya_qual
            bq_suffix = ""
            if include_barcode_quals and base_q is not None:
                qt = []
                bq_str = str(base_q)
                for bc_col, (bs, be) in bc_starts_ends.items():
                    if bs is not None and be is not None and be > bs:
                        qt.append(f"{bc_col}:{bq_str[bs:be]}")
                if us is not None and ue is not None and ue > us:
                    qt.append(f"UMI:{bq_str[us:ue]}")
                if qt:
                    bq_suffix = f"|BQ:{';'.join(qt)}"
            header = (
                f"@{read_name}_{cell_id_str}{umi_name} "
                f"cell_id:{cell_id_str}|Barcodes:{bc_str}{umi_field}|orientation:{orientation}"
                f"{bq_suffix}"
            )
            demux_record = f"{header}\n{seq_out}\n+\n{qual}\n"
        else:
            header = (
                f">{read_name}_{cell_id_str}{umi_name} "
                f"cell_id:{cell_id_str}|Barcodes:{bc_str}{umi_field}|orientation:{orientation}"
            )
            demux_record = f"{header}\n{seq_out}\n"

    return correction, demux_record, is_ambiguous


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
    max_queue_size=3,
    include_barcode_quals=False,
    include_polya=False,
    run_demux=False,
    keep_demux_chunk_outputs_after_combine=False,
    resume=True,
    checkpoint_file=None,
):
    """Orchestrate barcode correction on annotated reads with optional demux.

    Architecture: a scanner thread pre-reads polars chunks into a queue.
    The main thread dispatches rows to Pool workers for combined correction +
    demux record building, then writes TSV + demux gzip per chunk.
    Supports checkpoint/resume for crash recovery.
    """
    from multiprocessing import Pool

    import polars as pl

    from scripts.barcode_correction import (
        _combine_demux_chunk_outputs,
        _has_usable_base_qualities,
        _infer_barcode_columns,
    )
    from scripts.trained_models import seq_orders

    os.makedirs(output_dir, exist_ok=True)

    multi_file_input = isinstance(input_file, list)

    if multi_file_input:
        # Multi-file input: scan chunk files directly (whitelist-free fast path)
        logger.info(f"Scanning {len(input_file)} chunk files directly (skipping intermediate combine).")
        has_parquet = any(f.endswith(".parquet") for f in input_file)
        if has_parquet:
            lazy_df = pl.scan_parquet(input_file)
        else:
            lazy_df = pl.scan_csv(input_file, separator="\t", infer_schema_length=5000)
    else:
        if input_file is None:
            metadata_dir = f"{input_dir}/annotation_metadata"
            annotation_default = f"{metadata_dir}/annotations_valid.parquet"
            corrected_default = f"{metadata_dir}/annotations_valid_bc_corrected.parquet"
            if os.path.exists(annotation_default):
                input_file = annotation_default
            elif os.path.exists(corrected_default):
                input_file = corrected_default
            else:
                input_file = annotation_default
        if not os.path.exists(input_file):
            # Fallback: prior run may have replaced annotations_valid with bc_corrected
            metadata_dir = f"{input_dir}/annotation_metadata"
            corrected_default = f"{metadata_dir}/annotations_valid_bc_corrected.parquet"
            if os.path.exists(corrected_default):
                logger.info(f"Input file not found, using corrected fallback: {corrected_default}")
                input_file = corrected_default
            else:
                raise FileNotFoundError(f"Annotation file not found: {input_file}")

        if input_file.endswith(".parquet"):
            lazy_df = pl.scan_parquet(input_file)
        else:
            lazy_df = pl.scan_csv(input_file, separator="\t", infer_schema_length=5000)

    whitelist_df = pd.read_csv(whitelist_file, sep="\t")

    effective_output_fmt = output_fmt
    if run_demux and output_fmt == "fastq":
        if not _has_usable_base_qualities(lazy_df):
            logger.warning("Base quality scores not available; requested FASTQ demux output will be written as FASTA.")
            effective_output_fmt = "fasta"

    first_chunk = lazy_df.slice(0, chunk_size).collect()
    if first_chunk.height == 0:
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
    whitelist_sets = {bc: set(whitelist_dict[bc]) for bc in barcode_columns}

    # Pre-build cell_id lookup for single-barcode (vectorized assignment)
    if len(barcode_columns) == 1:
        bc_col = barcode_columns[0]
        cell_id_lookup = {seq: idx + 1 for idx, seq in whitelist_df[bc_col].items()}
    else:
        cell_id_lookup = None

    bc_metadata_dir = os.path.join(output_dir, "annotation_metadata")
    os.makedirs(bc_metadata_dir, exist_ok=True)
    corrected_parquet = f"{bc_metadata_dir}/annotations_valid_bc_corrected.parquet"
    in_place_corrected_overwrite = (
        not multi_file_input and os.path.abspath(input_file) == os.path.abspath(corrected_parquet)
    )
    corrected_parquet_tmp = (
        f"{corrected_parquet}.tmp" if in_place_corrected_overwrite else corrected_parquet
    )
    if os.path.exists(corrected_parquet_tmp):
        os.remove(corrected_parquet_tmp)
    if os.path.exists(corrected_parquet) and not in_place_corrected_overwrite:
        os.remove(corrected_parquet)

    # Per-chunk TSV output directory
    chunk_tsv_dir = os.path.join(bc_metadata_dir, "bc_correction_chunks")
    if os.path.exists(chunk_tsv_dir):
        shutil.rmtree(chunk_tsv_dir)
    os.makedirs(chunk_tsv_dir)

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

    # --- Checkpoint setup ---
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    if checkpoint_file is None:
        checkpoint_file = os.path.join(checkpoints_dir, "bc_correction_checkpoint.txt")

    resume_from_chunk = 0
    if resume and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as fh:
                parts = fh.readline().strip().split("\t")
            if len(parts) >= 1:
                resume_from_chunk = int(parts[0])
                logger.info(f"Checkpoint loaded: last completed chunk={resume_from_chunk}")
        except (ValueError, OSError):
            logger.warning("Could not parse bc correction checkpoint; starting from beginning.")
            resume_from_chunk = 0
    elif not resume and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    # Scanner thread pre-reads polars chunks; main thread processes them.
    chunk_queue = queue.Queue(maxsize=max_queue_size)
    scanner_error = {"exc": None}

    def _run_scanner():
        try:
            offset = 0
            chunk_idx = 0
            while True:
                chunk = lazy_df.slice(offset, chunk_size).collect()
                if chunk.height == 0:
                    break
                chunk_idx += 1
                chunk_queue.put((chunk_idx, chunk))
                logger.info(f"Scanner: queued chunk {chunk_idx}")
                offset += chunk_size
            chunk_queue.put(None)
            logger.info(f"Scanner: all {chunk_idx} chunks queued")
        except Exception as e:
            scanner_error["exc"] = e
            chunk_queue.put(None)

    scanner_thread = threading.Thread(target=_run_scanner, daemon=True)
    scanner_thread.start()

    mode = "correction + demux" if run_demux else "correction only (fast path)"
    logger.info(f"Starting barcode correction: {mode}, {threads} threads, queue prefetch={max_queue_size}")

    # Persistent Pool for fuzzy correction
    pool = Pool(threads) if threads > 1 else None
    chunk_paths = []
    total_chunks = 0

    try:
        while True:
            item = chunk_queue.get()
            if item is None:
                break

            chunk_idx, chunk = item  # polars DataFrame
            total_chunks = chunk_idx
            n_rows = chunk.height

            # --- Resume: skip already-completed chunks ---
            if resume_from_chunk > 0 and chunk_idx <= resume_from_chunk:
                chunk_path = os.path.join(chunk_tsv_dir, f"chunk{chunk_idx:06d}.tsv")
                if os.path.exists(chunk_path):
                    chunk_paths.append(chunk_path)
                    logger.info(f"Skipping chunk {chunk_idx} (already completed)")
                    continue

            # --- Pre-parse coordinates in polars (vectorized) ---
            def _pc(col_name):
                return (
                    pl.col(col_name).cast(pl.Utf8).str.split(",").list.first()
                    .str.strip_chars().replace(["", "None", "nan", "NaN"], None)
                    .cast(pl.Int64, strict=False)
                )

            coord_exprs = [_pc("cDNA_Starts").alias("_cs"), _pc("cDNA_Ends").alias("_ce")]
            has_umi = "UMI_Starts" in chunk.columns and "UMI_Ends" in chunk.columns
            if has_umi:
                coord_exprs += [_pc("UMI_Starts").alias("_us"), _pc("UMI_Ends").alias("_ue")]
            if include_polya:
                for col, alias in [("polyA_Starts", "_pas"), ("polyA_Ends", "_pae"),
                                   ("polyT_Starts", "_pts"), ("polyT_Ends", "_pte")]:
                    if col in chunk.columns:
                        coord_exprs.append(_pc(col).alias(alias))
            if include_barcode_quals:
                for bc in barcode_columns:
                    for sfx, a in [("_Starts", "_bs"), ("_Ends", "_be")]:
                        col = f"{bc}{sfx}"
                        if col in chunk.columns:
                            coord_exprs.append(_pc(col).alias(f"_{bc}{a}"))
            chunk = chunk.with_columns(coord_exprs)

            # --- Extract per-row data as lists ---
            reads = chunk["read"].to_list()
            read_names = chunk["ReadName"].to_list() if "ReadName" in chunk.columns else ["read"] * n_rows
            orientations = chunk["orientation"].to_list() if "orientation" in chunk.columns else ["NA"] * n_rows
            cs_list = chunk["_cs"].to_list()
            ce_list = chunk["_ce"].to_list()
            us_list = chunk["_us"].to_list() if has_umi else [None] * n_rows
            ue_list = chunk["_ue"].to_list() if has_umi else [None] * n_rows
            base_q_list = chunk["base_qualities"].to_list() if (effective_output_fmt == "fastq" and "base_qualities" in chunk.columns) else [None] * n_rows

            # Barcode sequences
            bc_seq_lists = {}
            for bc in barcode_columns:
                col = f"{bc}_Sequences"
                bc_seq_lists[bc] = chunk[col].to_list() if col in chunk.columns else [None] * n_rows

            # PolyA coordinates (coalesce polyA/polyT)
            pa_s_list = [None] * n_rows
            pa_e_list = [None] * n_rows
            if include_polya:
                _pas = chunk["_pas"].to_list() if "_pas" in chunk.columns else [None] * n_rows
                _pts = chunk["_pts"].to_list() if "_pts" in chunk.columns else [None] * n_rows
                _pae = chunk["_pae"].to_list() if "_pae" in chunk.columns else [None] * n_rows
                _pte = chunk["_pte"].to_list() if "_pte" in chunk.columns else [None] * n_rows
                pa_s_list = [a if a is not None else b for a, b in zip(_pas, _pts)]
                pa_e_list = [a if a is not None else b for a, b in zip(_pae, _pte)]

            # Barcode coordinate lists (for barcode quals)
            bc_starts_ends_lists = {}
            if include_barcode_quals:
                for bc in barcode_columns:
                    bs_col = f"_{bc}_bs"
                    be_col = f"_{bc}_be"
                    bs = chunk[bs_col].to_list() if bs_col in chunk.columns else [None] * n_rows
                    be = chunk[be_col].to_list() if be_col in chunk.columns else [None] * n_rows
                    bc_starts_ends_lists[bc] = (bs, be)

            # --- Build row_data tuples ---
            row_data_list = []
            for i in range(n_rows):
                bc_seqs = {bc: bc_seq_lists[bc][i] for bc in barcode_columns}
                bc_se = {}
                if include_barcode_quals:
                    for bc in barcode_columns:
                        bs, be = bc_starts_ends_lists[bc]
                        bc_se[bc] = (bs[i], be[i])
                row_data_list.append((
                    read_names[i], str(reads[i]) if reads[i] is not None else "",
                    orientations[i], bc_seqs,
                    cs_list[i], ce_list[i], us_list[i], ue_list[i],
                    base_q_list[i], pa_s_list[i], pa_e_list[i], bc_se,
                ))

            # --- Dispatch to Pool: combined correction + demux ---
            shared_args = (
                barcode_columns, whitelist_dict, whitelist_sets,
                cell_id_lookup, whitelist_df, bc_lv_threshold, strand, effective_output_fmt,
                include_barcode_quals, include_polya, run_demux,
            )
            task_args = [(rd,) + shared_args for rd in row_data_list]

            if pool is not None:
                results = pool.starmap(_correct_and_demux_row, task_args)
            else:
                results = [_correct_and_demux_row(*a) for a in task_args]

            n_fuzzy = sum(1 for corr, _, _ in results if any(
                corr.get(f"corrected_{bc}_min_dist", 0) not in (0, -1) for bc in barcode_columns
            ))
            logger.info(f"Chunk {chunk_idx}: {n_rows - n_fuzzy} exact/RC matches, {n_fuzzy} fuzzy ({100*(n_rows-n_fuzzy)//n_rows}%/{100*n_fuzzy//n_rows}%)")

            # --- Collect results: correction TSV ---
            correction_dicts = [r[0] for r in results]
            corr_df = pl.DataFrame(correction_dicts)

            # Merge correction columns back onto original chunk columns needed for TSV
            # Keep original annotation columns + add correction + cell_id columns
            keep_cols = [c for c in chunk.columns if not c.startswith("_")]
            out_df = chunk.select(keep_cols).hstack(corr_df)

            chunk_path = os.path.join(chunk_tsv_dir, f"chunk{chunk_idx:06d}.tsv")
            out_df.write_csv(chunk_path, separator="\t")
            chunk_paths.append(chunk_path)

            # --- Collect results: demux records ---
            if run_demux:
                import gzip as gzip_mod

                chunk_demuxed = os.path.join(demux_chunk_dir, f"chunk{chunk_idx:06d}.{demux_ext}.gz")
                chunk_ambiguous = os.path.join(ambiguous_chunk_dir, f"chunk{chunk_idx:06d}.{demux_ext}.gz")
                demux_lines = []
                amb_lines = []
                for _, demux_record, is_amb in results:
                    if demux_record is None:
                        continue
                    if is_amb:
                        amb_lines.append(demux_record)
                    else:
                        demux_lines.append(demux_record)
                with gzip_mod.open(chunk_demuxed, "wb", compresslevel=1) as fh:
                    fh.write("".join(demux_lines).encode())
                with gzip_mod.open(chunk_ambiguous, "wb", compresslevel=1) as fh:
                    fh.write("".join(amb_lines).encode())

            # Save checkpoint after chunk is fully written
            with open(checkpoint_file, "w") as cp_fh:
                cp_fh.write(f"{chunk_idx}\t{chunk_size}\n")

            logger.info(f"Corrected chunk {chunk_idx}")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    scanner_thread.join()
    if scanner_error["exc"]:
        raise scanner_error["exc"]

    if run_demux:
        _combine_demux_chunk_outputs(
            demux_chunk_dir,
            ambiguous_chunk_dir,
            demuxed_fasta,
            ambiguous_fasta,
            demux_ext,
            keep_demux_chunk_outputs_after_combine,
        )
        # Clean up demux chunk directories when not keeping
        if not keep_demux_chunk_outputs_after_combine:
            shutil.rmtree(demux_chunk_dir, ignore_errors=True)
            shutil.rmtree(ambiguous_chunk_dir, ignore_errors=True)

    # Combine chunk TSVs into final parquet
    logger.info(f"Combining {total_chunks} chunk TSVs into final parquet")
    if chunk_paths:
        pl.scan_csv(chunk_paths, separator="\t", infer_schema_length=5000).sink_parquet(
            corrected_parquet_tmp, compression="snappy"
        )

    # Cleanup
    shutil.rmtree(chunk_tsv_dir, ignore_errors=True)
    if in_place_corrected_overwrite:
        os.replace(corrected_parquet_tmp, corrected_parquet)
    if multi_file_input:
        # Clean up individual chunk files after successful correction
        if os.path.exists(corrected_parquet):
            for chunk_file in input_file:
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            logger.info(f"Cleaned up {len(input_file)} chunk files after correction.")
    elif os.path.basename(input_file) == "annotations_valid.parquet" and os.path.exists(input_file):
        if os.path.exists(corrected_parquet):
            os.remove(input_file)
    # Remove legacy TSV if present from prior runs
    legacy_tsv = f"{bc_metadata_dir}/annotations_valid_bc_corrected.tsv"
    if os.path.exists(legacy_tsv):
        os.remove(legacy_tsv)

    logger.info(f"Wrote corrected annotations to {corrected_parquet}")
