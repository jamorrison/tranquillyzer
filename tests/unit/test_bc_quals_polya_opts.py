import pandas as pd
import pytest
import sys
import types
import yaml

from scripts.correct_barcodes import process_row
from scripts.demultiplex import assign_cell_id
from scripts.trained_models import seq_orders
from wrappers.annotate_reads_wrap import annotate_reads_wrap

# Stub optional training-time deps so unit tests don't require them.
sys.modules.setdefault("tensorflow_addons", types.SimpleNamespace())
tf2crf_stub = types.ModuleType("tf2crf")
tf2crf_stub.CRF = object()
tf2crf_stub.ModelWithCRFLoss = object()
sys.modules.setdefault("tf2crf", tf2crf_stub)


def make_whitelist():
    df = pd.DataFrame([{"i7": "AAAA", "i5": "GGGG", "CBC": "CCCC"}])
    whitelist_dict = {
        "cell_ids": {1: "AAAA-GGGG-CCCC"},
        "i7": ["AAAA"],
        "i5": ["GGGG"],
        "CBC": ["CCCC"],
    }
    return df, whitelist_dict


def test_process_row_appends_polya_and_bq():
    whitelist_df, whitelist_dict = make_whitelist()
    row = {
        "ReadName": "r1",
        "read_length": 20,
        "read": "AAAAGGGGCCCCAAAATTTT",
        "cDNA_Starts": "0",
        "cDNA_Ends": "12",
        "UMI_Starts": "0",
        "UMI_Ends": "4",
        "random_s_Starts": "",
        "random_s_Ends": "",
        "random_e_Starts": "",
        "random_e_Ends": "",
        "polyA_Starts": "12",
        "polyA_Ends": "20",
        "architecture": "valid",
        "reason": "",
        "orientation": "+",
        "i7_Sequences": "AAAA",
        "i7_Starts": "0",
        "i7_Ends": "4",
        "i5_Sequences": "GGGG",
        "i5_Starts": "4",
        "i5_Ends": "8",
        "CBC_Sequences": "CCCC",
        "CBC_Starts": "8",
        "CBC_Ends": "12",
        "base_qualities": "ABCDEFGHIJKLMNOPQRST",
    }

    result, batch_reads = process_row(
        row,
        strand="fwd",
        barcode_columns=["i7", "i5", "CBC"],
        whitelist_dict=whitelist_dict,
        whitelist_df=whitelist_df,
        threshold=0,
        output_dir=".",
        output_fmt="fastq",
        include_barcode_quals_in_header=True,
        include_polya_in_output=True,
    )

    corrected_key = whitelist_dict["cell_ids"][result["cell_id"]]
    cell_reads = batch_reads[corrected_key]
    assert len(cell_reads) == 1
    header, seq_out, qual_out = cell_reads[0]

    assert header.startswith("@r1_AAAA-GGGG-CCCC_")
    assert "|BQ:i7:ABCD;i5:EFGH;CBC:IJKL;UMI:ABCD" in header
    assert seq_out == "AAAAGGGGCCCCAAAATTTT"  # cDNA + polyA tail
    assert qual_out == "ABCDEFGHIJKLMNOPQRST"


def test_annotate_reads_wrap_missing_model_lists_available(tmp_path):
    seq_orders_file = tmp_path / "seq_orders.yaml"
    config = {
        "model_a": {
            "strand": "fwd",
            "barcodes": ["i7"],
            "umis": ["UMI"],
            "segments": [{"name": "x", "pattern": "y"}],
        },
        "model_b": {
            "strand": "rev",
            "barcodes": ["i7"],
            "umis": ["UMI"],
            "segments": [{"name": "x", "pattern": "y"}],
        },
    }
    seq_orders_file.write_text(yaml.dump(config))
    whitelist_file = tmp_path / "whitelist.tsv"
    whitelist_file.write_text("i7\ni7_seq\n")

    with pytest.raises(ValueError) as excinfo:
        annotate_reads_wrap(
            output_dir=tmp_path,
            whitelist_file=str(whitelist_file),
            output_fmt="fastq",
            model_name="missing_model",
            model_type="CRF",
            seq_order_file=str(seq_orders_file),
            chunk_size=10,
            gpu_mem=None,
            target_tokens=10,
            vram_headroom=0.1,
            min_batch_size=1,
            max_batch_size=2,
            bc_lv_threshold=1,
            threads=1,
            max_queue_size=1,
            include_barcode_quals=False,
            include_polya=False,
        )

    msg = str(excinfo.value)
    assert "missing_model" in msg
    assert "model_a" in msg and "model_b" in msg


def test_assign_cell_id_supports_dynamic_multi_barcode_columns():
    whitelist_df = pd.DataFrame([{"i5": "GGGG", "i7": "AAAA", "cbc": "CCCC"}])
    row = {
        "corrected_i5": "GGGG",
        "corrected_i7": "AAAA",
        "corrected_cbc": "CCCC",
    }

    cell_id = assign_cell_id(row, whitelist_df, ["i5", "i7", "cbc"])

    assert cell_id == 1


def test_seq_orders_strips_whitespace_in_barcode_fields(tmp_path):
    seq_orders_file = tmp_path / "seq_orders.yaml"
    config = {
        "model_a": {
            "strand": "fwd",
            "barcodes": ["i5", "i7", "cbc"],
            "umis": ["UMI"],
            "segments": [
                {"name": "5p", "pattern": "A"},
                {"name": "i5", "pattern": "B"},
                {"name": "i7", "pattern": "C"},
                {"name": "cbc", "pattern": "D"},
                {"name": "cDNA", "pattern": "E"},
            ],
        }
    }
    seq_orders_file.write_text(yaml.dump(config))

    _, _, barcodes, umis, strand = seq_orders(str(seq_orders_file), "model_a")

    assert barcodes == ["i5", "i7", "cbc"]
    assert umis == ["UMI"]
    assert strand == "fwd"
