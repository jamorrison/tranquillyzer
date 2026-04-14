import os
import subprocess
import shutil
from pathlib import Path
import pytest

RAW_INPUT_DIR = Path("tests/10x3p/data")
OUT_DIR = Path("tests/10x3p")
SIM_DIR = Path("tests/10x3p")
BARCODES = Path("tests/10x3p/barcodes.tsv")
REF_FASTA = Path("tests/references/hg38_gencode_chr21.fa")
REF_GTF = Path("tests/references/hg38_gencode_chr21.gtf")
transcriptome = Path("tests/references/")
MODELS_DIR = Path("models")
THREADS = 1

OUT_DIR.mkdir(exist_ok=True, parents=True)

COVRC = str(Path(".coveragerc").resolve())
COVDATA = str(Path(".coverage").resolve())


def run_cmd(cmd, timeout=900, expect_code=0):
    env = os.environ.copy()
    env["COVERAGE_PROCESS_START"] = COVRC
    env["COVERAGE_FILE"] = COVDATA
    env.setdefault("PYTHONPATH", os.getcwd())

    print(f"\n>> Running: {' '.join(map(str, cmd))}")
    p = subprocess.run(
        list(map(str, cmd)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    print(p.stdout)
    print(p.stderr)
    assert p.returncode == expect_code, f"Command failed ({p.returncode}): {' '.join(map(str, cmd))}"
    return p


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_outputs():
    keep_names = {"barcodes.tsv", "data", "test_pipeline.tsv", "test_pipeline.py"}
    for item in OUT_DIR.iterdir():
        if item.name in keep_names:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


@pytest.mark.order(1)
def test_preprocess_wo_base_qual():
    run_cmd(
        [
            "tranquillyzer",
            "preprocess",
            RAW_INPUT_DIR,
            OUT_DIR,
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(2)
def test_readlengthdist():
    run_cmd(
        [
            "tranquillyzer",
            "readlengthdist",
            OUT_DIR,
        ]
    )


@pytest.mark.order(3)
def test_visualize():
    run_cmd(
        [
            "tranquillyzer",
            "visualize",
            OUT_DIR,
            "--models-dir",
            MODELS_DIR,
            "--num-reads",
            "10",
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(4)
def test_annotate_reads():
    demux_dir = OUT_DIR / "demuxed_fasta"
    if demux_dir.exists():
        shutil.rmtree(demux_dir)

    run_cmd(
        [
            "tranquillyzer",
            "annotate-reads",
            OUT_DIR,
            "--models-dir",
            MODELS_DIR,
            "--chunk-size",
            100000,
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(5)
def test_generate_whitelist():
    run_cmd(
        [
            "tranquillyzer",
            "generate-whitelist",
            OUT_DIR,
        ]
    )


@pytest.mark.order(6)
def test_demux_reads_bulk_fasta():
    # Must run before barcode-correct, which deletes annotations_valid.parquet.
    run_cmd(
        [
            "tranquillyzer",
            "demux-reads",
            OUT_DIR,
            "--input-file",
            f"{OUT_DIR}/annotation_metadata/annotations_valid.parquet",
            "--output-fmt",
            "fasta",
        ]
    )


@pytest.mark.order(7)
def test_barcode_correct():
    run_cmd(
        [
            "tranquillyzer",
            "barcode-correct",
            OUT_DIR,
            BARCODES,
            "--run-demux",
            "--output-fmt",
            "fastq",
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(8)
def test_demux_reads():
    run_cmd(
        [
            "tranquillyzer",
            "demux-reads",
            OUT_DIR,
            "--output-fmt",
            "fastq",
        ]
    )


@pytest.mark.order(9)
def test_qc_metrics_basic():
    run_cmd(
        [
            "tranquillyzer",
            "qc-metrics",
            OUT_DIR,
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(10)
def test_align():
    run_cmd(
        [
            "tranquillyzer",
            "align",
            OUT_DIR,
            REF_FASTA,
            OUT_DIR,
            "--preset",
            "splice",
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(11)
def test_dedup():
    run_cmd(
        [
            "tranquillyzer",
            "dedup",
            OUT_DIR,
            "--lv-threshold",
            1,
            "--threads",
            THREADS,
            "--per-cell",
        ]
    )


@pytest.mark.order(12)
def test_split_bam():
    run_cmd(
        [
            "tranquillyzer",
            "split-bam",
            f"{OUT_DIR}/aligned_files/demuxed_aligned_dup_marked.bam",
            "--bucket-threads",
            1,
            "--merge-threads",
            1,
            "--max-open-cb-writers",
            500,
            "--filter-secondary",
            "--filter-supplementary",
        ]
    )


@pytest.mark.order(13)
def test_featurecounts():
    run_cmd(
        [
            "tranquillyzer",
            "featurecounts",
            f"{OUT_DIR}/aligned_files/split_bams",
            REF_GTF,
            f"{OUT_DIR}/featurecounts_out",
            "--bind",
            "/varidata:/varidata",
            "--extra",
            "-t exon -g gene_id -O -s 2",
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(14)
def test_qc_metrics_with_bam():
    run_cmd(
        [
            "tranquillyzer",
            "qc-metrics",
            OUT_DIR,
            "--bam",
            f"{OUT_DIR}/aligned_files/demuxed_aligned_dup_marked.bam",
            "--counts-matrix",
            f"{OUT_DIR}/featurecounts_out/counts_matrix.tsv",
            "--gtf",
            REF_GTF,
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(15)
def test_simulate_data():
    run_cmd(
        [
            "tranquillyzer",
            "simulate-data",
            "template_model",
            SIM_DIR,
            "--num-reads",
            1000,
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(16)
def test_train_model():
    run_cmd(
        [
            "tranquillyzer",
            "train-model",
            "template_model",
            OUT_DIR,
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(17)
def test_assess_model():
    # Use a dedicated output dir so assess-model's annotation_metadata
    # doesn't collide with the main pipeline's parquet (which has reads
    # named read1/read2/..., not assess_*, causing the segment-metric
    # branch to skip everything).
    run_cmd(
        [
            "tranquillyzer",
            "assess-model",
            "template_model",
            f"{OUT_DIR}/template_model",
            f"{OUT_DIR}/assess_out",
            "--num-reads",
            200,
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(18)
def test_available_models():
    run_cmd(
        [
            "tranquillyzer",
            "availablemodels",
        ]
    )


@pytest.mark.order(19)
def test_available_gpus():
    run_cmd(
        [
            "tranquillyzer",
            "available-gpus",
        ]
    )


@pytest.mark.order(20)
def test_preprocess_w_base_qual():
    run_cmd(
        [
            "tranquillyzer",
            "preprocess",
            RAW_INPUT_DIR,
            OUT_DIR,
            "--output-base-qual",
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(21)
def test_annotate_reads_w_inline_bc_demux():
    run_cmd(
        [
            "tranquillyzer",
            "annotate-reads",
            OUT_DIR,
            "--whitelist-file",
            BARCODES,
            "--output-fmt",
            "fastq",
            "--models-dir",
            MODELS_DIR,
            "--run-barcode-correction",
            "--run-demux",
            "--chunk-size",
            100000,
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(22)
def test_demux_reads_after_inline_demux():
    # Re-emit reads from the corrected parquet produced by the inline-demux
    # annotate-reads run above. Exercises the demux-columns branch in
    # demux_wrap (_write_from_demux_columns).
    run_cmd(
        [
            "tranquillyzer",
            "demux-reads",
            OUT_DIR,
            "--output-fmt",
            "fastq",
        ]
    )
