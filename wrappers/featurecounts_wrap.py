"""Wrapper: run featureCounts (containerized) over per-cell BAMs and emit a counts matrix."""

import logging

logger = logging.getLogger(__name__)


def featurecounts_wrap(
    bam_dir,
    gtf,
    out_dir,
    container_runtime="auto",
    container_image=None,
    image_cache=None,
    extra_binds=None,
    batch_size=200,
    threads=8,
    workers=1,
    extra="-t exon -g gene_id -O",
    matrix_name="counts_matrix.tsv",
    no_run=False,
):
    """Resolve container runtime, pull image if needed, then run the batched featureCounts pipeline."""
    import os

    from scripts.container_runtime import DEFAULT_FEATURECOUNTS_IMAGE, resolve
    from scripts.featurecounts_matrix import run_featurecounts_matrix

    bam_dir = os.path.abspath(bam_dir)
    gtf = os.path.abspath(gtf)
    out_dir = os.path.abspath(out_dir)

    image = container_image or DEFAULT_FEATURECOUNTS_IMAGE
    bind_paths = [bam_dir, os.path.dirname(gtf), out_dir]
    if extra_binds:
        bind_paths.extend(p for p in extra_binds.split(",") if p.strip())
    rt, fc_cmd = resolve(
        tool="featureCounts",
        image=image,
        runtime=container_runtime,
        image_cache=image_cache,
        bind_paths=bind_paths,
    )
    logger.info("featureCounts runtime: %s", rt)
    logger.info("featureCounts invocation: %s", fc_cmd)

    return run_featurecounts_matrix(
        bam_dir=bam_dir,
        gtf=gtf,
        out_dir=out_dir,
        featurecounts=fc_cmd,
        batch_size=batch_size,
        threads=threads,
        extra=extra,
        matrix_name=matrix_name,
        workers=workers,
        no_run=no_run,
    )
