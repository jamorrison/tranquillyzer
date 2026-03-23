# Helper functions for annotate_reads_wrap.
# Verbatim from annotate_reads_wrap.py — only moved here.

import logging
import queue
import time
import os
import shutil
import re

logger = logging.getLogger(__name__)



def _has_usable_base_qualities_in_parquets(parquet_files, pl, sample_rows=5000):
    """Check if any parquet file contains non-null base quality scores."""
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


CHUNK_FILE_RE = re.compile(
    r"^pass(?P<pass>\d+)__(?P<bin>.+)__chunk(?P<chunk>\d{6})\.(?:tsv|done|parquet|fasta|fastq)(?:\.gz)?$"
)


def _save_checkpoint(checkpoint_file, pass_num, bin_name, chunk_idx, chunk_size):
    """Write checkpoint state (pass, bin, chunk, chunk_size) to file."""
    with open(checkpoint_file, "w") as fh:
        fh.write(f"{pass_num}\t{bin_name}\t{int(chunk_idx)}\t{int(chunk_size)}\n")


def _load_checkpoint(checkpoint_file, expected_chunk_size=None):
    """Load and validate checkpoint from file, returning (pass, bin, chunk) or None."""
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
    """Extract a sortable (pass, bin_order, bin, chunk) key from a chunk filename."""
    m = CHUNK_FILE_RE.match(name)
    if not m:
        return None
    pass_num = int(m.group("pass"))
    bin_name = m.group("bin")
    chunk_idx = int(m.group("chunk"))
    return pass_num, bin_order.get(bin_name, 10**9), bin_name, chunk_idx


def _done_marker_path(chunk_output_dir, pass_num, bin_name, chunk_idx):
    """Return the path to the done-marker file for a given chunk."""
    return os.path.join(chunk_output_dir, "done", f"pass{pass_num}__{bin_name}__chunk{int(chunk_idx):06d}.done")


def _cleanup_from_checkpoint(chunk_output_dir, checkpoint_tuple, bin_order):
    """Remove chunk output files at or after the checkpoint position."""
    if checkpoint_tuple is None:
        return
    cp_pass, cp_bin, cp_chunk = checkpoint_tuple
    cp_key = (cp_pass, bin_order.get(cp_bin, 10**9), cp_bin, cp_chunk)
    for subdir in ["done", "valid_chunks", "invalid_chunks", "demuxed_chunks", "ambiguous_chunks"]:
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
    """Remove all annotation output files and directories for a clean restart."""
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
        os.path.join(output_dir, "annotations_valid_bc_corrected.parquet"),
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
    run_barcode_correction,
    pl,
    chunk_size,
):
    """Convert chunk TSV outputs to Parquet and optionally combine into final files."""
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
    valid_output_name = "annotations_valid_bc_corrected.parquet" if run_barcode_correction else "annotations_valid.parquet"

    if combine_chunks:
        logger.info(
            f"Starting chunk combination: valid={len(valid_files)} TSV ({len(valid_chunk_parquets)} parquet), "
            f"invalid={len(invalid_files)} TSV ({len(invalid_chunk_parquets)} parquet)."
        )

        def _combine_valid_invalid(tsv_files, parquet_files, out_path, out_dir):
            """Combine chunk files (TSV, parquet, or mixed) into a single output parquet."""
            if tsv_files and parquet_files:
                # Mixed resume: convert stale TSVs to parquet, then combine all parquets
                _write_chunk_parquets(tsv_files, out_dir)
                all_parquets = _sorted_chunk_files(out_dir, ".parquet")
                try:
                    pl.scan_parquet(all_parquets).sink_parquet(
                        out_path, compression="snappy", row_group_size=chunk_size
                    )
                except Exception:
                    logger.warning("Parallel parquet scan failed, falling back to diagonal_relaxed concat.")
                    pl.concat(
                        [pl.scan_parquet(f) for f in all_parquets], how="diagonal_relaxed"
                    ).sink_parquet(out_path, compression="snappy", row_group_size=chunk_size)
                for tsv_path in tsv_files:
                    os.remove(tsv_path)
            elif tsv_files:
                # Legacy: all TSV
                pl.scan_csv(tsv_files, separator="\t", infer_schema_length=5000).sink_parquet(
                    out_path, compression="snappy", row_group_size=chunk_size
                )
                if keep_chunk_tsv_after_combine:
                    _write_chunk_parquets(tsv_files, out_dir)
                for tsv_path in tsv_files:
                    os.remove(tsv_path)
            elif parquet_files:
                # Parallel multi-file scan — Polars reads chunks across cores internally
                try:
                    pl.scan_parquet(parquet_files).sink_parquet(
                        out_path, compression="snappy", row_group_size=chunk_size
                    )
                except Exception:
                    # Fallback for schema mismatches (e.g., resumed runs with mixed formats)
                    logger.warning("Parallel parquet scan failed, falling back to diagonal_relaxed concat.")
                    pl.concat(
                        [pl.scan_parquet(f) for f in parquet_files], how="diagonal_relaxed"
                    ).sink_parquet(out_path, compression="snappy", row_group_size=chunk_size)

        _combine_valid_invalid(
            valid_files, valid_chunk_parquets, f"{output_dir}/{valid_output_name}", valid_out_dir
        )
        _combine_valid_invalid(
            invalid_files, invalid_chunk_parquets, f"{output_dir}/annotations_invalid.parquet", invalid_out_dir
        )
        logger.info(f"Finished chunk combination into {valid_output_name} and annotations_invalid.parquet.")

        if not keep_chunk_tsv_after_combine:
            # Re-scan for parquet chunks and delete them after successful combine
            for pq_path in _sorted_chunk_files(valid_out_dir, ".parquet") + _sorted_chunk_files(
                invalid_out_dir, ".parquet"
            ):
                os.remove(pq_path)
    else:
        # No-combine mode: ensure chunks are parquet (convert any legacy TSVs)
        if valid_files or invalid_files:
            logger.info(
                f"Starting TSV-to-parquet conversion for chunk outputs: valid={len(valid_files)}, "
                f"invalid={len(invalid_files)}."
            )
            _write_chunk_parquets(valid_files, valid_out_dir)
            _write_chunk_parquets(invalid_files, invalid_out_dir)
            logger.info("Finished TSV-to-parquet conversion for chunk outputs.")
            for tsv_path in valid_files + invalid_files:
                os.remove(tsv_path)


def _combine_demux_chunk_outputs(
    chunk_output_dir,
    output_dir,
    output_fmt,
    keep_demux_chunk_outputs_after_combine,
):
    """Concatenate per-chunk demux gzip files into final demuxed/ambiguous outputs."""
    ext = "fastq" if output_fmt == "fastq" else "fasta"
    demux_chunk_dir = os.path.join(chunk_output_dir, "demuxed_chunks")
    ambiguous_chunk_dir = os.path.join(chunk_output_dir, "ambiguous_chunks")
    final_dir = os.path.join(output_dir, "demuxed_fasta")
    os.makedirs(final_dir, exist_ok=True)

    def _sorted_chunk_files(path):
        if not os.path.isdir(path):
            return []
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(f".{ext}.gz")]
        files.sort(key=lambda f: _chunk_key_from_filename(os.path.basename(f), {}) or (10**9, 10**9, "", 10**9))
        return files

    def _concatenate_binary(files, out_path):
        if not files:
            return False
        with open(out_path, "wb") as out_fh:
            for src in files:
                with open(src, "rb") as src_fh:
                    shutil.copyfileobj(src_fh, out_fh)
        return True

    demux_files = _sorted_chunk_files(demux_chunk_dir)
    amb_files = _sorted_chunk_files(ambiguous_chunk_dir)
    demux_out = os.path.join(final_dir, f"demuxed.{ext}.gz")
    amb_out = os.path.join(final_dir, f"ambiguous.{ext}.gz")

    logger.info(
        f"Starting demux chunk combination: demuxed_chunks={len(demux_files)}, ambiguous_chunks={len(amb_files)}."
    )
    wrote_demux = _concatenate_binary(demux_files, demux_out)
    wrote_amb = _concatenate_binary(amb_files, amb_out)
    logger.info("Finished demux chunk combination.")

    if not keep_demux_chunk_outputs_after_combine:
        for chunk_file in demux_files + amb_files:
            os.remove(chunk_file)
        logger.info("Deleted demux chunk gzip members after successful demux combination.")

    return demux_out if wrote_demux else None, amb_out if wrote_amb else None


def load_libs():
    """Lazily import and return libraries needed by the annotation pipeline."""
    import os
    import gc
    import sys
    import resource
    import pickle
    import pandas as pd
    import multiprocessing as mp
    import psutil
    import polars as pl

    from filelock import FileLock

    from scripts.export_annotations import (
        post_process_reads,
    )
    from scripts.annotate_new_data import (
        calculate_total_rows,
        model_predictions,
        estimate_average_read_length_from_bin,
    )
    from scripts.preprocess_reads import convert_tsv_to_parquet
    from scripts.trained_models import seq_orders, get_valid_structures
    from scripts.available_gpus import log_gpus_used

    return (
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


