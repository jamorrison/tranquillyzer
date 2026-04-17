"""
featureCounts batching + per-cell counts matrix builder.

Vendored from tranquillyzer-nf/bin/featurecount_mtx.py and exposed as a
plain function (no Typer decorator) so the tranquillyzer wrapper can
call it directly.
"""

import glob
import logging
import multiprocessing
import os
import re
import shlex
import subprocess
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_bams(bam_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(bam_dir, "*.bam")), key=natural_key)


def run_featurecounts_batch(
    featurecounts: str,
    gtf: str,
    out_txt: str,
    bam_paths: List[str],
    threads: int,
    extra_args: List[str],
    retry_threads: List[int] = None,
    bisect_on_fail: bool = True,
    min_batch_size: int = 1,
) -> None:
    """Run featureCounts on a list of BAMs with retry + bisect on failure."""
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    if retry_threads is None:
        retry_threads = [threads, 16, 8, 4, 1]
        retry_threads = [t for t in retry_threads if t <= threads]
        retry_threads = list(dict.fromkeys(retry_threads))

    def _run_once(t, out_path, bams):
        cmd = shlex.split(featurecounts) + ["-a", gtf, "-o", out_path, "-T", str(t)] + extra_args + bams
        logger.info("Running featureCounts: %s", " ".join(shlex.quote(x) for x in cmd))
        return subprocess.run(cmd)

    for t in retry_threads:
        proc = _run_once(t, out_txt, bam_paths)
        if proc.returncode == 0:
            logger.info("featureCounts succeeded: out=%s (threads=%d, bams=%d)", out_txt, t, len(bam_paths))
            return
        logger.error(
            "featureCounts failed (rc=%d) out=%s threads=%d bams=%d",
            proc.returncode,
            out_txt,
            t,
            len(bam_paths),
        )

    if not bisect_on_fail or len(bam_paths) <= min_batch_size:
        raise subprocess.CalledProcessError(
            returncode=proc.returncode,
            cmd=[featurecounts, "-a", gtf, "-o", out_txt, "-T", str(retry_threads[-1])] + extra_args + bam_paths,
        )

    mid = len(bam_paths) // 2
    left, right = bam_paths[:mid], bam_paths[mid:]
    left_out = out_txt.replace(".txt", ".left.txt")
    right_out = out_txt.replace(".txt", ".right.txt")
    logger.warning("Bisecting batch due to repeated failure: %s -> (%d, %d)", out_txt, len(left), len(right))
    run_featurecounts_batch(
        featurecounts, gtf, left_out, left, threads, extra_args, retry_threads, bisect_on_fail, min_batch_size
    )
    run_featurecounts_batch(
        featurecounts, gtf, right_out, right, threads, extra_args, retry_threads, bisect_on_fail, min_batch_size
    )
    with open(out_txt + ".BISected", "w") as f:
        f.write("This batch was bisected into:\n")
        f.write(left_out + "\n")
        f.write(right_out + "\n")


def _run_batch_wrapper(kwargs):
    run_featurecounts_batch(**kwargs)


def parse_featurecounts_counts(path: str) -> Tuple[List[str], Dict[str, List[str]]]:
    sample_names: List[str] = []
    counts_by_gene: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            if line.startswith("Geneid\t"):
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 7:
                    raise ValueError(f"Unexpected featureCounts header in {path}")
                sample_names = cols[6:]
                continue
            cols = line.rstrip("\n").split("\t")
            gene = cols[0]
            counts = cols[6:]
            if sample_names and len(counts) != len(sample_names):
                raise ValueError(f"Sample column mismatch in {path} for gene {gene}")
            counts_by_gene[gene] = counts
    if not sample_names:
        raise ValueError(f"Did not find header in featureCounts file: {path}")
    return sample_names, counts_by_gene


def write_matrix(out_matrix, gene_order, all_samples, merged_counts):
    os.makedirs(os.path.dirname(out_matrix) or ".", exist_ok=True)
    with open(out_matrix, "w", encoding="utf-8") as out:
        out.write("Geneid\t" + "\t".join(all_samples) + "\n")
        for g in gene_order:
            out.write(g + "\t" + "\t".join(merged_counts[g]) + "\n")


def run_featurecounts_matrix(
    bam_dir: str,
    gtf: str,
    out_dir: str,
    featurecounts: str = "featureCounts",
    batch_size: int = 200,
    threads: int = 8,
    extra: str = "-t exon -g gene_id -O",
    matrix_name: str = "counts_matrix.tsv",
    workers: int = 1,
    no_run: bool = False,
) -> str:
    """Run featureCounts over many per-cell BAMs and merge into a gene×cell matrix.

    Returns the path to the merged matrix.
    """
    if not os.path.isdir(bam_dir):
        raise FileNotFoundError(f"bam_dir is not a directory: {bam_dir}")
    if not os.path.exists(gtf):
        raise FileNotFoundError(f"gtf not found: {gtf}")

    os.makedirs(out_dir, exist_ok=True)
    batches_dir = os.path.join(out_dir, "batches")
    os.makedirs(batches_dir, exist_ok=True)

    bams = find_bams(bam_dir)
    if not bams:
        raise FileNotFoundError(f"No BAMs found in: {bam_dir}")

    logger.info("BAMs found: %d", len(bams))
    extra_args = shlex.split(extra)

    batch_outputs: List[str] = []
    if not no_run:
        threads_per_worker = max(1, threads // workers)
        batch_args: List[dict] = []

        for i in range(0, len(bams), batch_size):
            chunk = bams[i : i + batch_size]
            batch_idx = i // batch_size
            out_txt = os.path.join(batches_dir, f"featurecounts_batch{batch_idx:04d}.txt")
            batch_outputs.append(out_txt)

            if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
                logger.info("skip (exists): %s", out_txt)
                continue

            batch_args.append(
                dict(
                    featurecounts=featurecounts,
                    gtf=gtf,
                    out_txt=out_txt,
                    bam_paths=chunk,
                    threads=threads_per_worker,
                    extra_args=extra_args,
                )
            )

        if batch_args:
            logger.info(
                "Running %d batches with %d workers (%d threads each)",
                len(batch_args),
                workers,
                threads_per_worker,
            )
            if workers > 1:
                with multiprocessing.Pool(workers) as pool:
                    pool.map(_run_batch_wrapper, batch_args)
            else:
                for kw in batch_args:
                    _run_batch_wrapper(kw)
    else:
        batch_outputs = sorted(glob.glob(os.path.join(batches_dir, "featurecounts_batch*.txt")), key=natural_key)
        if not batch_outputs:
            raise FileNotFoundError(f"--no-run specified but no batch outputs found in: {batches_dir}")

    logger.info("Batch outputs to merge: %d", len(batch_outputs))

    all_samples: List[str] = []
    gene_order: List[str] = []
    merged_counts: Dict[str, List[str]] = {}
    first = True

    for path in batch_outputs:
        samples, counts_by_gene = parse_featurecounts_counts(path)
        samples_norm = [os.path.splitext(os.path.basename(s))[0] for s in samples]

        if first:
            all_samples.extend(samples_norm)
            gene_order = list(counts_by_gene.keys())
            for g in gene_order:
                merged_counts[g] = counts_by_gene[g]
            first = False
        else:
            start_col = len(all_samples)
            all_samples.extend(samples_norm)
            for g in gene_order:
                prev = merged_counts[g]
                add = counts_by_gene.get(g)
                if add is None:
                    prev.extend(["0"] * len(samples_norm))
                else:
                    prev.extend(add)
            for g in counts_by_gene.keys():
                if g not in merged_counts:
                    merged_counts[g] = (["0"] * start_col) + counts_by_gene[g]
                    gene_order.append(g)

    out_matrix = os.path.join(out_dir, matrix_name)
    write_matrix(out_matrix, gene_order, all_samples, merged_counts)

    logger.info("Wrote matrix: %s", out_matrix)
    logger.info("Shape: genes=%d cells=%d", len(gene_order), len(all_samples))
    return out_matrix
