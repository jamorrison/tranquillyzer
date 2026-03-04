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
            "--output-file",
            "test_visualization",
            "--model-type",
            "CRF",
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
            "--model-type",
            "CRF",
            "--models-dir",
            MODELS_DIR,
            "--chunk-size",
            100000,
            "--threads",
            THREADS,
        ]
    )
    assert not demux_dir.exists(), "annotate-reads created demuxed_fasta even though demux was not requested"

@pytest.mark.order(5)
def test_barcode_correct():
    run_cmd(
        [
            "tranquillyzer",
            "barcode-correct",
            OUT_DIR,
            BARCODES,
            # "--output-fmt",
            # "fasta",
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(6)
def test_demux_reads():
    run_cmd(
        [
            "tranquillyzer",
            "demux-reads",
            OUT_DIR,
            "--input-file",
            f"{OUT_DIR}/annotations_valid_bc_corrected.parquet",
            "--output-fmt",
            "fasta",
        ]
    )


@pytest.mark.order(7)
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


@pytest.mark.order(8)
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


@pytest.mark.order(9)
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


# @pytest.mark.order(8)
# def test_simulate_data():
#     run_cmd(
#         [
#             "tranquillyzer",
#             "simulate-data",
#             "10x3p_sc_ont",
#             SIM_DIR,
#             "--num-reads",
#             1000,
#             "--threads",
#             THREADS,
#         ]
#     )


# @pytest.mark.order(9)
# def test_available_models():
#     run_cmd(
#         [
#             "tranquillyzer",
#             "availablemodels",
#         ]
#     )


# @pytest.mark.order(10)
# def test_preprocess_w_base_qual():
#     run_cmd(
#         [
#             "tranquillyzer",
#             "preprocess",
#             RAW_INPUT_DIR,
#             OUT_DIR,
#             "--output-base-qual",
#             "--threads",
#             THREADS,
#         ]
#     )


# @pytest.mark.order(11)
# def test_annotate_reads_w_base_qual():
#     run_cmd(
#         [
#             "tranquillyzer",
#             "annotate-reads",
#             OUT_DIR,
#             "--whitelist-file",
#             BARCODES,
#             "--output-fmt",
#             "fastq",
#             "--model-type",
#             "CRF",
#             "--models-dir",
#             MODELS_DIR,
#             "--run-barcode-correction",
#             "--run-demux",
#             "--chunk-size",
#             100000,
#             "--threads",
#             THREADS,
#         ]
#     )





# @pytest.mark.order(14)
# def test_annotate_reads_requires_whitelist_for_barcode_correction():
#     p = run_cmd(
#         [
#             "tranquillyzer",
#             "annotate-reads",
#             OUT_DIR,
#             "--model-type",
#             "CRF",
#             "--models-dir",
#             MODELS_DIR,
#             "--run-barcode-correction",
#             "--chunk-size",
#             100000,
#             "--threads",
#             THREADS,
#         ],
#         expect_code=2,
#     )
#     assert "whitelist_file is required" in p.stderr


# @pytest.mark.order(15)
# def test_train_model():
#     run_cmd(
#         [
#             "tranquillyzer",
#             "train-model",
#             "10x3p_sc_ont",
#             SIM_DIR,
#             "--threads",
#             THREADS,
#         ]
#     )


# @pytest.mark.order(16)
# def test_available_gpus():
#     run_cmd(
#         [
#             "tranquillyzer",
#             "available-gpus",
#         ]
#     )
