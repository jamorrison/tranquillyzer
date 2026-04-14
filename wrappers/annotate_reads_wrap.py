# Global library imports - used by all functions in module
import logging
import queue
import time

# Share logger across all functions in module
logger = logging.getLogger(__name__)


def annotate_reads_wrap(
    output_dir,
    whitelist_file,
    output_fmt,
    model_name,
    seq_order_file,
    chunk_size,
    gpu_mem,
    target_tokens,
    token_cap_above,
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
    split_concatenated=False,
):
    """Orchestrate the multi-pass annotation pipeline with optional barcode correction and demux."""
    from scripts.annotate_reads import (
        load_libs,
        _empty_results_queue,
        _has_usable_base_qualities_in_parquets,
        _save_checkpoint,
        _load_checkpoint,
        _done_marker_path,
        _cleanup_from_checkpoint,
        _cleanup_annotation_outputs_for_fresh_start,
        _convert_chunk_outputs,
        _combine_demux_chunk_outputs,
    )
    from scripts.annotate_new_data import load_model_for_inference

    (
        os,
        gc,
        sys,
        resource,
        pickle,
        mp,
        psutil,
        pl,
        FileLock,
        pd,
        model_predictions,
        post_process_reads,
        seq_orders,
        get_valid_structures,
        estimate_average_read_length_from_bin,
        calculate_total_rows,
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
        seq_order_file = os.path.join(utils_dir, "seq_orders.yaml")

    def _available_models(seq_orders_path):
        """Return list of model names defined in seq_orders.yaml (best-effort)."""
        import yaml

        models = []
        try:
            with open(seq_orders_path, "r") as f:
                config = yaml.safe_load(f)
            if isinstance(config, dict):
                models = list(config.keys())
        except Exception:
            pass
        return models

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

    model_path = f"{models_dir}/{model_name}.h5"
    with open(f"{models_dir}/{model_name}_lbl_bin.pkl", "rb") as f:
        label_binarizer = pickle.load(f)
    valid_structs = get_valid_structures(seq_order_file, model_name)

    # Build known_patterns for segments with literal adapter sequences
    import re as _re

    known_patterns = {}
    for seg_name, seg_pattern in zip(seq_order, sequences):
        if (
            seg_pattern not in ("NN", "RN", "A", "T")
            and not _re.match(r"N\d+", seg_pattern)
            and seg_name not in barcodes
        ):
            known_patterns[seg_name] = seg_pattern
    if known_patterns:
        logger.info(f"Known-pattern segments for edit distance: {list(known_patterns.keys())}")

    if run_barcode_correction:
        if not whitelist_file:
            raise ValueError("whitelist_file is required when run_barcode_correction=True")
        whitelist_df = pd.read_csv(whitelist_file, sep="\t")
    else:
        whitelist_df = pd.DataFrame()
    num_labels = len(seq_order)

    pp_base = preprocess_dir if preprocess_dir is not None else output_dir
    base_folder_path = os.path.join(pp_base, "full_length_pp_fa")

    chunk_output_dir = os.path.join(output_dir, "annotation_chunks")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_file = checkpoint_file or os.path.join(checkpoints_dir, "annotation_checkpoint.txt")

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
            logger.warning("Base quality scores not available; requested FASTQ demux output will be written as FASTA.")
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

    if run_demux:
        fasta_dir = os.path.join(output_dir, "demuxed_fasta")
        os.makedirs(fasta_dir, exist_ok=True)
        if effective_output_fmt == "fastq":
            logger.info("Selected demux output format: FASTQ")
        elif effective_output_fmt == "fasta":
            logger.info("Selected demux output format: FASTA")

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

                with FileLock(checkpoint_lock_path):
                    _save_checkpoint(checkpoint_file_worker, pass_num_worker, bin_name, chunk_idx, chunk_size)

                success = post_process_reads(
                    reads,
                    read_names,
                    strand,
                    output_fmt,
                    base_qualities,
                    pass_num_worker,
                    model_path,
                    predictions,
                    label_binarizer,
                    read_lengths,
                    seq_order,
                    bin_name,
                    chunk_idx,
                    output_dir,
                    barcodes,
                    whitelist_df,
                    whitelist_dict,
                    bc_lv_threshold,
                    threads,
                    include_barcode_quals,
                    include_polya,
                    run_barcode_correction,
                    run_demux,
                    chunk_output_dir,
                    split_concatenated,
                    valid_structs,
                    known_patterns,
                )

                if success:
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

    # Load the model once and reuse across all bins
    model, strategy, params, min_batch_from_model = load_model_for_inference(model_path, num_labels)
    if min_batch_size < min_batch_from_model:
        min_batch_size = min_batch_from_model

    task_queue = mp.Queue(maxsize=max_queue_size)
    result_queue = mp.Queue()
    count = mp.Value("i", 0)

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

    logger.info("Starting annotation pass on all reads")
    try:
        pass1_pointer = _resume_pointer_for_pass(1, parquet_files)
        if pass1_pointer is None:
            logger.info("Skipping annotation: checkpoint indicates it is already complete")
        else:
            start_bin_idx, start_chunk = pass1_pointer
            logger.info(
                f"Resume target -> start_bin={os.path.basename(run_files[0]).replace('.parquet', '') if (run_files := parquet_files[start_bin_idx:]) else 'N/A'}, "
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
                    model,
                    model_path,
                    strategy,
                    params,
                    num_labels,
                    user_total_gb=gpu_mem,
                    target_tokens_per_replica=target_tokens,
                    safety_margin=vram_headroom,
                    min_batch=min_batch_size,
                    max_batch=max_batch_size,
                    token_cap_above=token_cap_above,
                ):
                    task_queue.put(item)
                    queued_chunks += 1
            total_queued_chunks += queued_chunks
            if queued_chunks == 0:
                logger.info("No pending chunks found. Dataset is already annotated.")
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

    # Drain result queue
    idle_start = None
    while any(w.is_alive() for w in workers) or not result_queue.empty():
        try:
            result_queue.get(timeout=15)
            idle_start = None
        except queue.Empty:
            if idle_start is None:
                idle_start = time.time()
                logger.info("Result queue idle, waiting for worker results...")
            elif time.time() - idle_start > 60:
                raise TimeoutError("Result queue timed out after no data for 60 seconds and no workers finished.")

    logger.info(f"[Memory] RSS: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

    logger.info("Finished annotation pass on all reads")

    for worker in workers:
        worker.join()

    skip_chunk_output_conversion = False
    if resume and total_queued_chunks == 0:
        logger.info("All annotation chunks are already completed. Dataset has already been annotated.")
        if combine_chunk_outputs:
            valid_parquet_name = (
                "annotations_valid_bc_corrected.parquet" if run_barcode_correction else "annotations_valid.parquet"
            )
            metadata_dir = os.path.join(output_dir, "annotation_metadata")
            valid_parquet = os.path.join(metadata_dir, valid_parquet_name)
            invalid_parquet = os.path.join(metadata_dir, "annotations_invalid.parquet")
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
            run_barcode_correction,
            pl,
            chunk_size,
        )

    if run_demux:
        _combine_demux_chunk_outputs(
            chunk_output_dir,
            output_dir,
            effective_output_fmt,
            keep_demux_chunk_outputs_after_combine,
        )

    # Clean up annotation_chunks directory when not keeping chunks
    if not keep_chunk_tsv_after_combine:
        import shutil

        if os.path.isdir(chunk_output_dir):
            shutil.rmtree(chunk_output_dir, ignore_errors=True)

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    max_rss_mb = usage.ru_maxrss / 1024 if os.uname().sysname == "Linux" else usage.ru_maxrss  # Linux gives KB
    logger.info(f"Peak memory usage during annotation pipeline: {max_rss_mb:.2f} MB")
    logger.info(f"Elapsed time: {time.time() - start:.2f} seconds")
