# Global library imports - used by all functions in module
import logging
import queue
import time
import os
import gzip
import shutil
import re

# Share logger across all functions in module
logger = logging.getLogger(__name__)


def _gzip_file(path):
    if path is None or not isinstance(path, str) or not path.strip():
        return
    if not path.endswith((".fasta", ".fastq")):
        return
    if not os.path.exists(path):
        return
    gz_path = path + ".gz"
    if os.path.exists(gz_path):
        os.remove(gz_path)
    with open(path, "rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(path)


def _has_usable_base_qualities_in_parquets(parquet_files, pl, sample_rows=5000):
    for parquet_file in parquet_files:
        try:
            sample_df = pl.scan_parquet(parquet_file).limit(sample_rows).collect()
        except Exception:
            continue
        if "base_qualities" not in sample_df.columns:
            continue
        for value in sample_df["base_qualities"].to_list():
            if value is None:
                continue
            text = str(value).strip()
            if text and text.lower() not in {"none", "nan"}:
                return True
    return False


CHUNK_FILE_RE = re.compile(r"^pass(?P<pass>\d+)__(?P<bin>.+)__chunk(?P<chunk>\d{6})\.(?:tsv|done|parquet|fasta|fastq)$")


def _save_checkpoint(checkpoint_file, pass_num, bin_name, chunk_idx, chunk_size):
    with open(checkpoint_file, "w") as fh:
        fh.write(f"{pass_num}\t{bin_name}\t{int(chunk_idx)}\t{int(chunk_size)}\n")


def _load_checkpoint(checkpoint_file, expected_chunk_size=None):
    if not os.path.exists(checkpoint_file):
        return None
    with open(checkpoint_file, "r") as fh:
        raw = fh.readline().strip()
    if not raw:
        return None
    parts = raw.split("\t")
    if len(parts) not in {3, 4}:
        return None
    try:
        pass_num = int(parts[0])
        bin_name = parts[1]
        chunk_idx = int(parts[2])
    except ValueError:
        return None

    if len(parts) == 4:
        try:
            saved_chunk_size = int(parts[3])
        except ValueError:
            return None
    else:
        saved_chunk_size = None

    if saved_chunk_size is None and expected_chunk_size is not None:
        logger.warning(
            "Checkpoint is in legacy format without chunk size; cannot validate resume chunk_size consistency."
        )

    if (
        saved_chunk_size is not None
        and expected_chunk_size is not None
        and int(saved_chunk_size) != int(expected_chunk_size)
    ):
        raise ValueError(
            "Checkpoint chunk_size mismatch: checkpoint was created with chunk_size="
            f"{saved_chunk_size}, but current run uses chunk_size={int(expected_chunk_size)}. "
            "Use the same chunk_size or start with a fresh checkpoint/output directory."
        )

    return pass_num, bin_name, chunk_idx


def _chunk_key_from_filename(name, bin_order):
    m = CHUNK_FILE_RE.match(name)
    if not m:
        return None
    pass_num = int(m.group("pass"))
    bin_name = m.group("bin")
    chunk_idx = int(m.group("chunk"))
    return pass_num, bin_order.get(bin_name, 10**9), bin_name, chunk_idx


def _done_marker_path(chunk_output_dir, pass_num, bin_name, chunk_idx):
    return os.path.join(chunk_output_dir, "done", f"pass{pass_num}__{bin_name}__chunk{int(chunk_idx):06d}.done")


def _cleanup_from_checkpoint(chunk_output_dir, checkpoint_tuple, bin_order):
    if checkpoint_tuple is None:
        return
    cp_pass, cp_bin, cp_chunk = checkpoint_tuple
    cp_key = (cp_pass, bin_order.get(cp_bin, 10**9), cp_bin, cp_chunk)
    for subdir in ["done", "valid_chunks", "invalid_chunks"]:
        root = os.path.join(chunk_output_dir, subdir)
        if not os.path.isdir(root):
            continue
        for fn in os.listdir(root):
            key = _chunk_key_from_filename(fn, bin_order)
            if key is None:
                continue
            if key >= cp_key:
                os.remove(os.path.join(root, fn))


def _cleanup_annotation_outputs_for_fresh_start(output_dir, checkpoint_file):
    paths_to_remove = [
        os.path.join(output_dir, "annotation_chunks"),
        os.path.join(output_dir, "annotations_valid_chunks"),
        os.path.join(output_dir, "annotations_invalid_chunks"),
        os.path.join(output_dir, "demuxed_fasta"),
        os.path.join(output_dir, "tmp_invalid_reads"),
        os.path.join(output_dir, "annotations_valid.tsv"),
        os.path.join(output_dir, "annotations_invalid.tsv"),
        os.path.join(output_dir, "annotations_valid.tsv.lock"),
        os.path.join(output_dir, "annotations_invalid.tsv.lock"),
        os.path.join(output_dir, "annotations_valid.parquet"),
        os.path.join(output_dir, "annotations_invalid.parquet"),
        checkpoint_file,
        checkpoint_file + ".lock",
    ]
    for path in paths_to_remove:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except FileNotFoundError:
            continue


def _convert_chunk_outputs(
    chunk_output_dir,
    output_dir,
    combine_chunks,
    keep_chunk_tsv_after_combine,
    pl,
    chunk_size,
):
    def _sorted_chunk_files(path, suffix):
        if not os.path.isdir(path):
            return []
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(suffix)]
        files.sort(key=lambda f: _chunk_key_from_filename(os.path.basename(f), {}) or (10**9, 10**9, "", 10**9))
        return files

    def _write_chunk_parquets(tsv_files, parquet_out_dir):
        wrote_any = False
        os.makedirs(parquet_out_dir, exist_ok=True)
        for tsv_path in tsv_files:
            stem = os.path.splitext(os.path.basename(tsv_path))[0]
            pl.scan_csv(tsv_path, separator="\t", infer_schema_length=5000).sink_parquet(
                os.path.join(parquet_out_dir, f"{stem}.parquet"), compression="snappy", row_group_size=chunk_size
            )
            wrote_any = True
        return wrote_any

    valid_files = _sorted_chunk_files(os.path.join(chunk_output_dir, "valid_chunks"), ".tsv")
    invalid_files = _sorted_chunk_files(os.path.join(chunk_output_dir, "invalid_chunks"), ".tsv")
    valid_chunk_parquets = _sorted_chunk_files(os.path.join(chunk_output_dir, "valid_chunks"), ".parquet")
    invalid_chunk_parquets = _sorted_chunk_files(os.path.join(chunk_output_dir, "invalid_chunks"), ".parquet")
    valid_out_dir = os.path.join(chunk_output_dir, "valid_chunks")
    invalid_out_dir = os.path.join(chunk_output_dir, "invalid_chunks")

    if combine_chunks:
        logger.info(
            f"Starting chunk combination: valid={len(valid_files)} TSV ({len(valid_chunk_parquets)} parquet), "
            f"invalid={len(invalid_files)} TSV ({len(invalid_chunk_parquets)} parquet)."
        )
        if valid_files:
            pl.scan_csv(valid_files, separator="\t", infer_schema_length=5000).sink_parquet(
                f"{output_dir}/annotations_valid.parquet", compression="snappy", row_group_size=chunk_size
            )
        elif valid_chunk_parquets:
            pl.scan_parquet(valid_chunk_parquets).sink_parquet(
                f"{output_dir}/annotations_valid.parquet", compression="snappy", row_group_size=chunk_size
            )
        if invalid_files:
            pl.scan_csv(invalid_files, separator="\t", infer_schema_length=5000).sink_parquet(
                f"{output_dir}/annotations_invalid.parquet", compression="snappy", row_group_size=chunk_size
            )
        elif invalid_chunk_parquets:
            pl.scan_parquet(invalid_chunk_parquets).sink_parquet(
                f"{output_dir}/annotations_invalid.parquet", compression="snappy", row_group_size=chunk_size
            )
        logger.info("Finished chunk combination into annotations_valid.parquet and annotations_invalid.parquet.")
        if keep_chunk_tsv_after_combine:
            logger.info(
                f"Starting TSV-to-parquet conversion for chunk outputs: valid={len(valid_files)}, "
                f"invalid={len(invalid_files)}."
            )
            _write_chunk_parquets(valid_files, valid_out_dir)
            _write_chunk_parquets(invalid_files, invalid_out_dir)
            logger.info("Finished TSV-to-parquet conversion for chunk outputs.")
        if valid_files or invalid_files:
            for tsv_path in valid_files + invalid_files:
                os.remove(tsv_path)
    else:
        logger.info(
            f"Starting TSV-to-parquet conversion for chunk outputs: valid={len(valid_files)}, "
            f"invalid={len(invalid_files)}."
        )
        wrote_valid = _write_chunk_parquets(valid_files, valid_out_dir)
        wrote_invalid = _write_chunk_parquets(invalid_files, invalid_out_dir)
        logger.info("Finished TSV-to-parquet conversion for chunk outputs.")
        if wrote_valid or wrote_invalid:
            for tsv_path in valid_files + invalid_files:
                os.remove(tsv_path)


def _combine_demux_chunk_outputs(chunk_output_dir, output_dir, output_fmt, keep_demux_chunk_outputs_after_combine):
    ext = "fastq" if output_fmt == "fastq" else "fasta"
    demux_chunk_dir = os.path.join(chunk_output_dir, "demuxed_chunks")
    ambiguous_chunk_dir = os.path.join(chunk_output_dir, "ambiguous_chunks")
    final_dir = os.path.join(output_dir, "demuxed_fasta")
    os.makedirs(final_dir, exist_ok=True)

    def _sorted_chunk_files(path):
        if not os.path.isdir(path):
            return []
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(f".{ext}")]
        files.sort(key=lambda f: _chunk_key_from_filename(os.path.basename(f), {}) or (10**9, 10**9, "", 10**9))
        return files

    def _concatenate(files, out_path):
        if not files:
            return False
        with open(out_path, "w") as out_fh:
            for src in files:
                with open(src, "r") as src_fh:
                    shutil.copyfileobj(src_fh, out_fh)
        return True

    demux_files = _sorted_chunk_files(demux_chunk_dir)
    amb_files = _sorted_chunk_files(ambiguous_chunk_dir)
    demux_out = os.path.join(final_dir, f"demuxed.{ext}")
    amb_out = os.path.join(final_dir, f"ambiguous.{ext}")

    logger.info(
        f"Starting demux chunk combination: demuxed_chunks={len(demux_files)}, ambiguous_chunks={len(amb_files)}."
    )
    wrote_demux = _concatenate(demux_files, demux_out)
    wrote_amb = _concatenate(amb_files, amb_out)
    logger.info("Finished demux chunk combination.")

    if not keep_demux_chunk_outputs_after_combine:
        for chunk_file in demux_files + amb_files:
            os.remove(chunk_file)
        logger.info("Deleted demux chunk FASTA/FASTQ files after successful demux combination.")

    return demux_out if wrote_demux else None, amb_out if wrote_amb else None


def load_libs():
    import os
    import gc
    import sys
    import resource
    import pickle
    import pandas as pd
    import multiprocessing as mp
    from collections import defaultdict
    import psutil
    import polars as pl

    from filelock import FileLock

    from scripts.export_annotations import (
        post_process_reads,
        plot_read_n_cDNA_lengths,
    )
    from scripts.annotate_new_data import (
        calculate_total_rows,
        model_predictions,
        estimate_average_read_length_from_bin,
    )
    from scripts.preprocess_reads import convert_tsv_to_parquet
    from scripts.trained_models import seq_orders
    from scripts.available_gpus import log_gpus_used

    return (
        os,
        gc,
        sys,
        resource,
        pickle,
        mp,
        defaultdict,
        psutil,
        pl,
        FileLock,
        pd,
        model_predictions,
        post_process_reads,
        seq_orders,
        estimate_average_read_length_from_bin,
        calculate_total_rows,
        plot_read_n_cDNA_lengths,
        convert_tsv_to_parquet,
        log_gpus_used,
    )


def collect_prediction_stats(result_queue, workers, max_idle_time=60):
    """Collect and drain worker results to avoid queue buildup."""
    idle_start = None

    while any(worker.is_alive() for worker in workers) or not result_queue.empty():
        try:
            result = result_queue.get(timeout=15)

            # Reset idle start since a new result has been retrieved
            idle_start = None

            _ = result
        except queue.Empty:
            # Wait for the queue to get more entries
            if idle_start is None:
                idle_start = time.time()
                logger.info("Result queue idle, waiting for worker results...")
            elif time.time() - idle_start > max_idle_time:
                raise TimeoutError(
                    f"Result queue timed out after no data for {max_idle_time} seconds and no workers finished."
                )


def _empty_results_queue(result_queue, workers, max_idle_time=60):
    """Runs when error encountered during model prediction. Clears queue to allow for clean exit."""
    idle_start = None

    while any(worker.is_alive() for worker in workers) or not result_queue.empty():
        try:
            # We need to get the items in the queue to clear it, but we don't actually want to do anything
            _ = result_queue.get(timeout=15)

            # Reset idle start since we retrieved a new result
            idle_start = None
        except queue.Empty:
            # Wait for the queue to get more entries
            if idle_start is None:
                idle_start = time.time()
                logger.info("Result queue idle during shutdown, waiting for more results")
            elif time.time() - idle_start > max_idle_time:
                # This may be overkill
                raise TimeoutError(
                    f"Result queue timed out during shutdown after no data received for {max_idle_time} seconds with workers still running."
                )

    logger.info("Results queue cleared. Commence shutdown due to error")


def annotate_reads_wrap(
    output_dir,
    whitelist_file,
    output_fmt,
    model_name,
    model_type,
    seq_order_file,
    chunk_size,
    gpu_mem,
    target_tokens,
    vram_headroom,
    min_batch_size,
    max_batch_size,
    bc_lv_threshold,
    threads,
    max_queue_size,
    include_barcode_quals,
    include_polya,
    run_barcode_correction=False,
    run_demux=False,
    checkpoint_file=None,
    resume=True,
    combine_chunk_outputs=True,
    keep_chunk_tsv_after_combine=False,
    keep_demux_chunk_outputs_after_combine=False,
    models_dir=None,
    preprocess_dir=None,
):
    (
        os,
        gc,
        sys,
        resource,
        pickle,
        mp,
        defaultdict,
        psutil,
        pl,
        FileLock,
        pd,
        model_predictions,
        post_process_reads,
        seq_orders,
        estimate_average_read_length_from_bin,
        calculate_total_rows,
        plot_read_n_cDNA_lengths,
        convert_tsv_to_parquet,
        log_gpus_used,
    ) = load_libs()

    start = time.time()

    # Let user know whether they're running on CPU only or GPU (provided handles if so)
    log_gpus_used()

    # Read / create / prepare input files and directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(base_dir, ".."))
    if models_dir is None:
        models_dir = os.path.join(base_dir, "models")
    models_dir = os.path.abspath(models_dir)
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Model directory not found: {models_dir}")

    utils_dir = os.path.join(base_dir, "utils")
    utils_dir = os.path.abspath(utils_dir)

    if seq_order_file is None:
        seq_order_file = os.path.join(utils_dir, "seq_orders.tsv")

    def _available_models(seq_orders_path):
        """Return list of model names defined in seq_orders.tsv (best-effort)."""
        models = []
        try:
            with open(seq_orders_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    model_id = (line.strip().replace("'", "").replace('"', "").split("\t")[0]).strip()
                    if model_id:
                        models.append(model_id)
        except Exception:
            pass
        return models

    # TODO: model_path and model_path_w_CRF can probably be moved into the model if-statement
    #       This may have to wait until post_process_worker has been moved out of this function though
    model_path_w_CRF = None
    model_path = None

    if model_type == "REG" or model_type == "HYB":
        model_path = f"{models_dir}/{model_name}.h5"
        with open(f"{models_dir}/{model_name}_lbl_bin.pkl", "rb") as f:
            label_binarizer = pickle.load(f)

    try:
        seq_order, sequences, barcodes, UMIs, strand = seq_orders(seq_order_file, model_name)
    except Exception as e:
        available = _available_models(seq_order_file)
        suffix = f" Available models: {', '.join(available)}" if available else " No models found in seq_orders file."
        raise ValueError(f"Model '{model_name}' not found in seq_orders file: {seq_order_file}.{suffix}") from e
    if not seq_order:
        available = _available_models(seq_order_file)
        suffix = f" Available models: {', '.join(available)}" if available else " No models found in seq_orders file."
        raise ValueError(f"Model '{model_name}' not found in seq_orders file: {seq_order_file}.{suffix}")
    if run_barcode_correction:
        if not whitelist_file:
            raise ValueError("whitelist_file is required when run_barcode_correction=True")
        whitelist_df = pd.read_csv(whitelist_file, sep="\t")
    else:
        whitelist_df = pd.DataFrame()
    num_labels = len(seq_order)

    pp_base = preprocess_dir if preprocess_dir is not None else output_dir
    base_folder_path = os.path.join(pp_base, "full_length_pp_fa")

    invalid_output_file = os.path.join(output_dir, "annotations_invalid.tsv")
    valid_output_file = os.path.join(output_dir, "annotations_valid.tsv")
    chunk_output_dir = os.path.join(output_dir, "annotation_chunks")
    checkpoint_file = checkpoint_file or os.path.join(output_dir, "annotation_checkpoint.txt")

    if not resume:
        logger.info("Resume disabled. Removing existing annotation outputs and starting from scratch.")
        _cleanup_annotation_outputs_for_fresh_start(output_dir, checkpoint_file)

    os.makedirs(os.path.join(chunk_output_dir, "done"), exist_ok=True)
    os.makedirs(os.path.join(chunk_output_dir, "valid_chunks"), exist_ok=True)
    os.makedirs(os.path.join(chunk_output_dir, "invalid_chunks"), exist_ok=True)

    parquet_files = sorted(
        [
            os.path.join(base_folder_path, f)
            for f in os.listdir(base_folder_path)
            if f.endswith(".parquet") and not f.endswith("read_index.parquet")
        ],
        key=lambda f: estimate_average_read_length_from_bin(os.path.basename(f).replace(".parquet", "")),
    )
    bin_order = {os.path.basename(f).replace(".parquet", ""): i for i, f in enumerate(parquet_files)}
    checkpoint_tuple = _load_checkpoint(checkpoint_file, expected_chunk_size=chunk_size) if resume else None
    if checkpoint_tuple:
        cp_pass, cp_bin, cp_chunk = checkpoint_tuple
        cp_done = _done_marker_path(chunk_output_dir, cp_pass, cp_bin, cp_chunk)
        logger.info(
            f"Resume enabled. Checkpoint loaded: pass={cp_pass}, bin={cp_bin}, chunk={cp_chunk} "
            f"(status: {'done' if os.path.exists(cp_done) else 'incomplete'})"
        )
        if os.path.exists(cp_done):
            logger.info(
                f"Checkpointed chunk already completed. Will resume from next chunk after "
                f"pass={cp_pass}, bin={cp_bin}, chunk={cp_chunk}"
            )
        else:
            logger.info(f"Will resume from checkpointed chunk: pass={cp_pass}, bin={cp_bin}, chunk={cp_chunk}")
        # Only rewind outputs when checkpointed chunk is incomplete.
        # If its .done marker exists, keep outputs as-is and rely on done markers to skip work.
        if not os.path.exists(cp_done):
            _cleanup_from_checkpoint(chunk_output_dir, checkpoint_tuple, bin_order)
    elif resume:
        logger.info("Resume enabled but no checkpoint found. Starting from beginning.")

    effective_output_fmt = output_fmt
    if run_demux and output_fmt == "fastq":
        has_base_qualities = _has_usable_base_qualities_in_parquets(parquet_files, pl)
        if not has_base_qualities:
            logger.warning(
                "Base quality scores not available; requested FASTQ demux output will be written as FASTA."
            )
            effective_output_fmt = "fasta"
    if run_demux and not run_barcode_correction:
        logger.info("Bulk export mode enabled: demux output will contain all valid reads without barcode correction.")
    if run_demux and include_barcode_quals:
        if not run_barcode_correction:
            logger.warning(
                "--include-barcode-quals requested, but barcode correction is disabled; barcode quality tags will be omitted."
            )
        elif effective_output_fmt == "fastq":
            logger.info("Barcode quality strings will be appended to FASTQ headers for demuxed reads.")
        else:
            logger.warning(
                "--include-barcode-quals requested, but demux output format is FASTA; barcode quality tags will be omitted."
            )

    demuxed_fasta = None
    ambiguous_fasta = None
    demuxed_fasta_lock = None
    ambiguous_fasta_lock = None
    if run_demux:
        fasta_dir = os.path.join(output_dir, "demuxed_fasta")
        os.makedirs(fasta_dir, exist_ok=True)

        if effective_output_fmt == "fastq":
            logger.info("Selected demux output format: FASTQ")
            demuxed_fasta = os.path.join(fasta_dir, "demuxed.fastq")
            ambiguous_fasta = os.path.join(fasta_dir, "ambiguous.fastq")
        elif effective_output_fmt == "fasta":
            logger.info("Selected demux output format: FASTA")
            demuxed_fasta = os.path.join(fasta_dir, "demuxed.fasta")
            ambiguous_fasta = os.path.join(fasta_dir, "ambiguous.fasta")

    invalid_file_lock = FileLock(invalid_output_file + ".lock")
    valid_file_lock = FileLock(valid_output_file + ".lock")

    # TODO: This entire object could be dropped and use the barcodes list as it comes
    #       from seq_orders. There is only one location where both the key and value are
    #       used from the dictionary. Since its the same value though, we really don't
    #       need to double store these values
    column_mapping = {barcode: barcode for barcode in barcodes}
    if run_barcode_correction:
        whitelist_dict = {
            "cell_ids": {
                idx + 1: "-".join(map(str, row.dropna().unique()))
                for idx, row in whitelist_df[list(column_mapping.values())].iterrows()
            },
            **{
                input_column: whitelist_df[whitelist_column].dropna().unique().tolist()
                for input_column, whitelist_column in column_mapping.items()
            },
        }
    else:
        whitelist_dict = {"cell_ids": {}}

    def post_process_worker(
        task_queue,
        strand,
        output_fmt,
        count,
        header_track,
        result_queue,
        include_barcode_quals,
        include_polya,
        run_barcode_correction,
        run_demux,
        pass_num_worker,
        checkpoint_file_worker,
        checkpoint_lock_path,
    ):
        """Worker function for processing reads and returning results."""
        while True:
            try:
                item = task_queue.get(timeout=10)
                if item is None:
                    break

                parquet_file, bin_name, chunk_idx, predictions, read_names, reads, read_lengths, base_qualities = item

                with header_track.get_lock():
                    add_header = header_track.value == 0

                with FileLock(checkpoint_lock_path):
                    _save_checkpoint(checkpoint_file_worker, pass_num_worker, bin_name, chunk_idx, chunk_size)

                result = post_process_reads(
                    reads,
                    read_names,
                    strand,
                    output_fmt,
                    base_qualities,
                    model_type,
                    pass_num_worker,
                    model_path_w_CRF,
                    predictions,
                    label_binarizer,
                    {},
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
                    bc_lv_threshold,
                    checkpoint_file_worker,
                    1,
                    {},
                    {},
                    demuxed_fasta,
                    demuxed_fasta_lock,
                    ambiguous_fasta,
                    ambiguous_fasta_lock,
                    threads,
                    include_barcode_quals,
                    include_polya,
                    run_barcode_correction,
                    run_demux,
                    chunk_output_dir,
                )

                if result:
                    with FileLock(checkpoint_lock_path):
                        _save_checkpoint(checkpoint_file_worker, pass_num_worker, bin_name, chunk_idx, chunk_size)
                    result_queue.put((True, bin_name, chunk_idx))
                else:
                    logger.warning(f"No result from post_process_reads in {bin_name}, chunk {chunk_idx}")

                with count.get_lock():
                    count.value += 1

                gc.collect()
            except queue.Empty:
                pass

    num_workers = min(threads, mp.cpu_count() - 1)
    max_queue_size = max(3, num_workers * 2)
    total_queued_chunks = 0

    def _resume_pointer_for_pass(pass_num_local, bin_files):
        """Return (start_bin_index, start_chunk) or None if this pass is already complete."""
        if not resume or checkpoint_tuple is None:
            return 0, 1

        cp_pass, cp_bin, cp_chunk = checkpoint_tuple
        if cp_pass > pass_num_local:
            return None
        if cp_pass < pass_num_local:
            return 0, 1

        bin_names = [os.path.basename(p).replace(".parquet", "") for p in bin_files]
        if cp_bin not in bin_names:
            return 0, 1

        start_bin_idx = bin_names.index(cp_bin)
        start_chunk = cp_chunk
        cp_done = _done_marker_path(chunk_output_dir, cp_pass, cp_bin, cp_chunk)
        if os.path.exists(cp_done):
            start_chunk += 1
        return start_bin_idx, start_chunk

    if model_type == "REG" or model_type == "HYB":
        task_queue = mp.Queue(maxsize=max_queue_size)
        result_queue = mp.Queue()
        count = mp.Value("i", 0)
        header_track = mp.Value("i", 0)

        pass_num = 1

        logger.info(f"[Memory] RSS: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

        workers = [
            mp.Process(
                target=post_process_worker,
                args=(
                    task_queue,
                    strand,
                    effective_output_fmt,
                    count,
                    header_track,
                    result_queue,
                    include_barcode_quals,
                    include_polya,
                    run_barcode_correction,
                    run_demux,
                    pass_num,
                    checkpoint_file,
                    checkpoint_file + ".lock",
                ),
            )
            for _ in range(num_workers)
        ]

        logger.info(f"Number of workers = {len(workers)}")

        for worker in workers:
            worker.start()

        # process all the reads with CNN-LSTM model first
        logger.info("Starting first pass with regular model on all the reads")
        try:
            pass1_pointer = _resume_pointer_for_pass(1, parquet_files)
            if pass1_pointer is None:
                logger.info("Skipping pass 1: checkpoint indicates it is already complete")
            else:
                start_bin_idx, start_chunk = pass1_pointer
                logger.info(
                    f"Pass 1 resume target -> start_bin={os.path.basename(run_files[0]).replace('.parquet','') if (run_files := parquet_files[start_bin_idx:]) else 'N/A'}, "
                    f"start_chunk={start_chunk}"
                )
                run_files = parquet_files[start_bin_idx:]
                queued_chunks = 0
                for i, parquet_file in enumerate(run_files):
                    chunk_start_local = start_chunk if i == 0 else 1
                    for item in model_predictions(
                        parquet_file,
                        chunk_start_local,
                        chunk_size,
                        model_path,
                        model_path_w_CRF,
                        model_type,
                        num_labels,
                        user_total_gb=gpu_mem,
                        target_tokens_per_replica=target_tokens,
                        safety_margin=vram_headroom,
                        min_batch=min_batch_size,
                        max_batch=max_batch_size,
                    ):
                        task_queue.put(item)
                        with header_track.get_lock():
                            header_track.value += 1
                        queued_chunks += 1
                total_queued_chunks += queued_chunks
                if queued_chunks == 0:
                    logger.info("No pending chunks found for pass 1. This pass is already annotated.")
        except Exception as e:
            # Wind down queues, close workers when done, print error and exit
            for _ in range(threads):
                task_queue.put(None)

            _empty_results_queue(result_queue, workers)
            for worker in workers:
                worker.join()
                worker.close()

            logger.error(
                f"Error found while annotating: {e}. Resume from the last checkpoint by re-running with resume enabled. Exiting!"
            )
            sys.exit(1)

        logger.info(f"[Memory] RSS: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

        for _ in range(threads):
            task_queue.put(None)

        logger.info(f"[Memory] RSS: {psutil.Process().memory_info().rss / 1e6:.2f} MB")
        collect_prediction_stats(result_queue, workers)
        logger.info(f"[Memory] RSS: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

        logger.info("Finished first pass with regular model on all the reads")

        for worker in workers:
            worker.join()

        logger.info(f"[Memory] RSS: {psutil.Process().memory_info().rss / 1e6:.2f} MB")
        model_path_w_CRF = f"{models_dir}/{model_name}_w_CRF.h5"

        if model_type == "HYB":
            tmp_invalid_dir = os.path.join(output_dir, "tmp_invalid_reads")

            convert_tsv_to_parquet(tmp_invalid_dir, row_group_size=1000000)
            invalid_parquet_files = sorted(
                [
                    os.path.join(tmp_invalid_dir, f)
                    for f in os.listdir(tmp_invalid_dir)
                    if f.endswith(".parquet") and not f.endswith("read_index.parquet")
                ],
                key=lambda f: estimate_average_read_length_from_bin(os.path.basename(f).replace(".parquet", "")),
            )

            with open(f"{models_dir}/{model_name}_w_CRF_lbl_bin.pkl", "rb") as f:
                label_binarizer = pickle.load(f)

            # if model type selcted is HYB, process the failed reads in step 1 with CNN-LSTM-CRF model
            pass_num = 2
            task_queue = mp.Queue(maxsize=max_queue_size)
            result_queue = mp.Queue()

            with count.get_lock():
                count.value = 0

            with header_track.get_lock():
                header_track.value = 0

            workers = [
                mp.Process(
                    target=post_process_worker,
                    args=(
                        task_queue,
                        strand,
                        effective_output_fmt,
                        count,
                        header_track,
                        result_queue,
                        include_barcode_quals,
                        include_polya,
                        run_barcode_correction,
                        run_demux,
                        pass_num,
                        checkpoint_file,
                        checkpoint_file + ".lock",
                    ),
                )
                for _ in range(num_workers)
            ]

            for worker in workers:
                worker.start()

            logger.info("Starting second pass with CRF model on invalid reads")
            try:
                pass2_pointer = _resume_pointer_for_pass(2, invalid_parquet_files)
                if pass2_pointer is None:
                    logger.info("Skipping pass 2: checkpoint indicates it is already complete")
                else:
                    start_bin_idx, start_chunk = pass2_pointer
                    run_files = invalid_parquet_files[start_bin_idx:]
                    logger.info(
                        f"Pass 2 resume target -> start_bin={os.path.basename(run_files[0]).replace('.parquet','') if run_files else 'N/A'}, "
                        f"start_chunk={start_chunk}"
                    )
                    queued_chunks = 0
                    for i, invalid_parquet_file in enumerate(run_files):
                        if calculate_total_rows(invalid_parquet_file) < 100:
                            continue
                        chunk_start_local = start_chunk if i == 0 else 1
                        for item in model_predictions(
                            invalid_parquet_file,
                            chunk_start_local,
                            chunk_size,
                            model_path,
                            model_path_w_CRF,
                            model_type,
                            num_labels,
                            user_total_gb=gpu_mem,
                            target_tokens_per_replica=target_tokens,
                            safety_margin=vram_headroom,
                            min_batch=min_batch_size,
                            max_batch=max_batch_size,
                        ):
                            task_queue.put(item)
                            with header_track.get_lock():
                                header_track.value += 1
                            queued_chunks += 1
                    total_queued_chunks += queued_chunks
                    if queued_chunks == 0:
                        logger.info("No pending chunks found for pass 2. This pass is already annotated.")
            except Exception as e:
                # Wind down queues, close workers when done, print error and exit
                for _ in range(threads):
                    task_queue.put(None)

                _empty_results_queue(result_queue, workers)
                for worker in workers:
                    worker.join()
                    worker.close()

                logger.error(
                    f"Error found while annotating: {e}. Resume from the last checkpoint by re-running with resume enabled. Exiting!"
                )
                sys.exit(1)

            for _ in range(threads):
                task_queue.put(None)

            collect_prediction_stats(result_queue, workers)
            logger.info("Finished second pass with CRF model on invalid reads")

            for worker in workers:
                worker.join()

    if model_type == "CRF":
        # process all the reads with CNN-LSTM-CRF model
        model_path_w_CRF = f"{models_dir}/{model_name}_w_CRF.h5"

        with open(f"{models_dir}/{model_name}_w_CRF_lbl_bin.pkl", "rb") as f:
            label_binarizer = pickle.load(f)

        task_queue = mp.Queue(maxsize=max_queue_size)
        result_queue = mp.Queue()
        count = mp.Value("i", 0)
        header_track = mp.Value("i", 0)

        pass_num = 1

        workers = [
            mp.Process(
                target=post_process_worker,
                args=(
                    task_queue,
                    strand,
                    effective_output_fmt,
                    count,
                    header_track,
                    result_queue,
                    include_barcode_quals,
                    include_polya,
                    run_barcode_correction,
                    run_demux,
                    pass_num,
                    checkpoint_file,
                    checkpoint_file + ".lock",
                ),
            )
            for _ in range(num_workers)
        ]

        for worker in workers:
            worker.start()

        logger.info("Starting first pass with CRF model on all the reads")
        try:
            pass1_pointer = _resume_pointer_for_pass(1, parquet_files)
            if pass1_pointer is None:
                logger.info("Skipping CRF pass: checkpoint indicates it is already complete")
            else:
                start_bin_idx, start_chunk = pass1_pointer
                run_files = parquet_files[start_bin_idx:]
                logger.info(
                    f"CRF pass resume target -> start_bin={os.path.basename(run_files[0]).replace('.parquet','') if run_files else 'N/A'}, "
                    f"start_chunk={start_chunk}"
                )
                queued_chunks = 0
                for i, parquet_file in enumerate(run_files):
                    chunk_start_local = start_chunk if i == 0 else 1
                    for item in model_predictions(
                        parquet_file,
                        chunk_start_local,
                        chunk_size,
                        None,
                        model_path_w_CRF,
                        model_type,
                        num_labels,
                        user_total_gb=gpu_mem,
                        target_tokens_per_replica=target_tokens,
                        safety_margin=vram_headroom,
                        min_batch=min_batch_size,
                        max_batch=max_batch_size,
                    ):
                        task_queue.put(item)
                        with header_track.get_lock():
                            header_track.value += 1
                        queued_chunks += 1
                total_queued_chunks += queued_chunks
                if queued_chunks == 0:
                    logger.info("No pending chunks found for CRF pass. This pass is already annotated.")
        except Exception as e:
            # Wind down queues, close workers when done, print error and exit
            for _ in range(threads):
                task_queue.put(None)

            _empty_results_queue(result_queue, workers)
            for worker in workers:
                worker.join()
                worker.close()

            logger.error(
                f"Error found while annotating: {e}. Resume from the last checkpoint by re-running with resume enabled. Exiting!"
            )
            sys.exit(1)

        for _ in range(threads):
            task_queue.put(None)

        collect_prediction_stats(result_queue, workers)

        logger.info("Finished first pass with CRF model on all the reads")

        for worker in workers:
            worker.join()
            worker.close()

    skip_chunk_output_conversion = False
    if resume and total_queued_chunks == 0:
        logger.info("All annotation chunks are already completed. Dataset has already been annotated.")
        if combine_chunk_outputs:
            valid_parquet = os.path.join(output_dir, "annotations_valid.parquet")
            invalid_parquet = os.path.join(output_dir, "annotations_invalid.parquet")
            if os.path.exists(valid_parquet) and os.path.exists(invalid_parquet):
                logger.info(
                    "Skipping chunk combination because combined parquet outputs already exist "
                    "and no new chunks were processed."
                )
                skip_chunk_output_conversion = True

    if not skip_chunk_output_conversion:
        _convert_chunk_outputs(
            chunk_output_dir,
            output_dir,
            combine_chunk_outputs,
            keep_chunk_tsv_after_combine,
            pl,
            chunk_size,
        )

    if run_demux:
        demuxed_fasta, ambiguous_fasta = _combine_demux_chunk_outputs(
            chunk_output_dir,
            output_dir,
            effective_output_fmt,
            keep_demux_chunk_outputs_after_combine,
        )
        _gzip_file(demuxed_fasta)
        _gzip_file(ambiguous_fasta)

    if model_type == "HYB":
        tmp_invalid_dir = os.path.join(output_dir, "tmp_invalid_reads")
        if os.path.isdir(tmp_invalid_dir):
            shutil.rmtree(tmp_invalid_dir)

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    max_rss_mb = usage.ru_maxrss / 1024 if os.uname().sysname == "Linux" else usage.ru_maxrss  # Linux gives KB
    logger.info(f"Peak memory usage during annotation pipeline: {max_rss_mb:.2f} MB")
    logger.info(f"Elapsed time: {time.time() - start:.2f} seconds")
