import logging

logger = logging.getLogger(__name__)


def evaluate_model_wrap(
    model_name,
    model_dir,
    output_dir,
    seq_orders_file,
    num_reads,
    concat_fraction,
    mismatch_rate,
    insertion_rate,
    deletion_rate,
    min_cDNA,
    max_cDNA,
    polyT_error_rate,
    max_insertions,
    threads,
    rc,
    transcriptome,
    max_trunc_5p,
    max_trunc_3p,
    min_spacer,
    max_spacer,
    bin_size,
    max_read_length,
    gpu_mem,
    target_tokens,
    vram_headroom,
    min_batch_size,
    max_batch_size,
    resume=True,
):
    """Generate assessment reads, run annotation pipeline, and compute model metrics."""
    import glob
    import json
    import os
    import random

    import numpy as np
    import polars as pl
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    import multiprocessing

    import math

    import subprocess
    import sys

    from scripts.trained_models import seq_orders, get_assessment_structures
    from scripts.simulate_training_data import simulate_and_write_fasta
    from utils import get_version

    __version__ = get_version()

    # ── resolve paths ──

    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(base_dir, "..")
    utils_dir = os.path.abspath(os.path.join(base_dir, "utils"))

    if seq_orders_file is None:
        seq_orders_file = os.path.join(utils_dir, "seq_orders.yaml")
    if not os.path.exists(seq_orders_file):
        raise FileNotFoundError(f"Seq orders file not found: {seq_orders_file}")

    model_dir = os.path.abspath(model_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ── load model config ──

    seq_order, sequences, barcodes, UMIs, strand = seq_orders(seq_orders_file, model_name)
    assessment_structs = get_assessment_structures(seq_orders_file, model_name)

    # ── rescale proportions if concat_fraction specified ──

    if concat_fraction is not None:
        single_structs = [s for s in assessment_structs if s.get("repeat", 1) == 1]
        concat_structs = [s for s in assessment_structs if s.get("repeat", 1) > 1]

        if single_structs:
            single_prop = (1.0 - concat_fraction) / len(single_structs)
            for s in single_structs:
                s["proportion"] = single_prop
        if concat_structs:
            concat_prop = concat_fraction / len(concat_structs)
            for s in concat_structs:
                s["proportion"] = concat_prop

    # ── cap cDNA length per structure if max_read_length specified ──

    if max_read_length is not None:
        import re

        for s in assessment_structs:
            repeat = s.get("repeat", 1)
            # Compute fixed-length overhead per fragment from patterns
            fixed_per_frag = 0
            for pat in s["patterns"]:
                if re.match(r"N\d+", pat):
                    fixed_per_frag += int(pat[1:])
                elif pat in ("NN", "RN"):
                    pass  # variable length — cDNA or spacer
                elif pat in ("A", "T"):
                    fixed_per_frag += 25  # estimated polyA/T average
                else:
                    fixed_per_frag += len(pat)  # literal adapter

            # Spacer cDNA flanks: (repeat + 1) spacers at avg max_spacer/2
            spacer_overhead = (repeat + 1) * max_spacer
            total_fixed = fixed_per_frag * repeat + spacer_overhead
            available_for_cdna = max_read_length - total_fixed
            effective_max_cdna = max(min_cDNA, available_for_cdna // repeat)
            capped_max_cdna = min(max_cDNA, effective_max_cdna)
            s["length_range"] = (min_cDNA, capped_max_cdna)
            logger.info(
                f"Structure repeat={repeat}: capped max_cDNA from {max_cDNA} to {capped_max_cdna} "
                f"(max_read_length={max_read_length}, fixed_overhead={total_fixed})"
            )

    # ── define output paths ──

    length_range = (min_cDNA, max_cDNA)
    fasta_dir = os.path.join(output_dir, "assessment_fasta")
    gt_dir = os.path.join(output_dir, "assessment_gt")
    gt_parquet_path = os.path.join(gt_dir, "assessment_gt.parquet")
    read_index_path = os.path.join(output_dir, "full_length_pp_fa", "read_index.parquet")
    annot_dir = os.path.join(output_dir, "annotation_metadata")
    valid_parquet = os.path.join(annot_dir, "annotations_valid.parquet")
    invalid_parquet = os.path.join(annot_dir, "annotations_invalid.parquet")
    metrics_dir = os.path.join(output_dir, "assessment_metrics")
    report_path = os.path.join(metrics_dir, f"{model_name}_assessment_report.html")

    # ── Stage 1: Simulate assessment reads + save GT ──

    existing_fastas = glob.glob(os.path.join(fasta_dir, "*.fasta")) if os.path.isdir(fasta_dir) else []
    if resume and os.path.exists(gt_parquet_path) and existing_fastas:
        logger.info("Resuming: simulation and GT already exist, skipping")
        gt_df = pl.read_parquet(gt_parquet_path)
        all_labels = [json.loads(x) for x in gt_df["gt_labels"].to_list()]
        all_expected_fragments = gt_df["expected_fragments"].to_list()
        all_structure_names = gt_df["structure_type"].to_list()
    else:
        # Prepare transcriptome
        if transcriptome:
            logger.info("Loading transcriptome FASTA")
            transcriptome_records = list(SeqIO.parse(transcriptome, "fasta"))
            logger.info(f"Loaded {len(transcriptome_records)} transcripts")
        else:
            logger.info("Generating random transcripts for assessment reads")
            transcriptome_records = []
            for i in range(num_reads):
                length = random.randint(min_cDNA, max_cDNA)
                seq_str = "".join(np.random.choice(list("ATCG")) for _ in range(length))
                record = SeqRecord(
                    Seq(seq_str), id=f"random_transcript_{i + 1}", description=f"Synthetic transcript {i + 1}"
                )
                transcriptome_records.append(record)

        # Convert transcriptome to plain strings for fast pickling
        transcriptome_seqs = [str(rec.seq) if hasattr(rec, "seq") else str(rec) for rec in transcriptome_records]
        del transcriptome_records

        os.makedirs(fasta_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        # Each FASTA chunk = 4000 output reads -> 2000 pre-RC reads per chunk
        reads_per_chunk = 2000
        num_chunks = max(1, math.ceil(num_reads / reads_per_chunk))
        chunk_sizes = [reads_per_chunk] * num_chunks
        total_assigned = reads_per_chunk * num_chunks
        if total_assigned > num_reads:
            chunk_sizes[-1] -= (total_assigned - num_reads)

        worker_args = []
        start_idx = 0
        for chunk_idx, chunk_n_reads in enumerate(chunk_sizes):
            fasta_path = os.path.join(fasta_dir, f"assessment_reads_{chunk_idx}.fasta")
            worker_args.append((
                chunk_n_reads, length_range, mismatch_rate, insertion_rate, deletion_rate,
                polyT_error_rate, max_insertions, assessment_structs, transcriptome_seqs,
                rc, max_trunc_5p, max_trunc_3p, min_spacer, max_spacer, fasta_path, start_idx,
            ))
            start_idx += chunk_n_reads * (2 if rc else 1)

        total_expected = sum(cs * (2 if rc else 1) for cs in chunk_sizes)
        logger.info(
            f"Generating {num_reads} assessment reads (rc={rc}, total output={total_expected}) "
            f"across {num_chunks} chunks with up to {min(threads, num_chunks)} workers"
        )

        all_labels = []
        all_expected_fragments = []
        all_structure_names = []

        effective_workers = min(threads, num_chunks)
        if effective_workers > 1:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=effective_workers) as pool:
                for chunk_labels, chunk_frags, chunk_names in pool.imap(simulate_and_write_fasta, worker_args):
                    all_labels.extend(chunk_labels)
                    all_expected_fragments.extend(chunk_frags)
                    all_structure_names.extend(chunk_names)
        else:
            for args in worker_args:
                chunk_labels, chunk_frags, chunk_names = simulate_and_write_fasta(args)
                all_labels.extend(chunk_labels)
                all_expected_fragments.extend(chunk_frags)
                all_structure_names.extend(chunk_names)

        logger.info(f"Generated {len(all_labels)} assessment reads across {num_chunks} FASTA files")

        # Save GT metadata
        gt_records = []
        for i in range(len(all_labels)):
            gt_records.append({
                "ReadName": f"assess_{i}",
                "gt_labels": json.dumps(all_labels[i]),
                "expected_fragments": all_expected_fragments[i],
                "structure_type": all_structure_names[i],
            })
        gt_df = pl.DataFrame(gt_records)
        gt_df = gt_df.with_columns(pl.lit(__version__).alias("tranquillyzer_version"))
        gt_df = gt_df.with_columns(pl.lit(model_name).alias("model_name"))
        gt_df.write_parquet(gt_parquet_path)
        logger.info(f"Saved GT metadata for {len(all_labels)} reads")

    # ── Stage 2: Preprocess FASTA into length-binned parquets ──

    if resume and os.path.exists(read_index_path):
        logger.info("Resuming: preprocessing already done, skipping")
    else:
        logger.info("Preprocessing assessment reads into length-binned parquets")
        preprocess_cmd = [
            sys.executable, "-m", "main", "preprocess",
            fasta_dir, output_dir,
            "--bin-size", str(bin_size),
            "--threads", str(threads),
        ]
        subprocess.run(preprocess_cmd, check=True)

    # ── Stage 3: Run annotation pipeline ──

    if resume and os.path.exists(valid_parquet):
        logger.info("Resuming: annotation already done, skipping")
    else:
        logger.info("Running annotation pipeline on assessment reads")
        annotate_cmd = [
            sys.executable, "-m", "main", "annotate-reads",
            output_dir,
            "--model-name", model_name,
            "--models-dir", model_dir,
            "--preprocess-dir", output_dir,
            "--split-concatenated",
            "--threads", str(threads),
        ]
        if seq_orders_file:
            annotate_cmd += ["--seq-order-file", seq_orders_file]
        if gpu_mem:
            annotate_cmd += ["--gpu-mem", str(gpu_mem)]
        annotate_cmd += [
            "--target-tokens", str(target_tokens),
            "--vram-headroom", str(vram_headroom),
            "--min-batch-size", str(min_batch_size),
            "--max-batch-size", str(max_batch_size),
        ]
        subprocess.run(annotate_cmd, check=True)

    # ── Stage 4: Compute metrics ──

    if resume and os.path.exists(report_path):
        logger.info(f"Resuming: assessment report already exists at {report_path}, skipping")
    else:
        from scripts.evaluate_model import evaluate_model

        evaluate_model(
            gt_labels=all_labels,
            expected_fragments=all_expected_fragments,
            structure_names=all_structure_names,
            valid_parquet_path=valid_parquet,
            invalid_parquet_path=invalid_parquet,
            seq_order=seq_order,
            output_dir=metrics_dir,
            model_name=model_name,
        )

    logger.info(f"Assessment complete. Results in {metrics_dir}/")
