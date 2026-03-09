import pytest

from wrappers.annotate_reads_wrap import _load_checkpoint, _save_checkpoint


def test_checkpoint_roundtrip_persists_chunk_size(tmp_path):
    checkpoint_file = tmp_path / "annotation_checkpoint.txt"

    _save_checkpoint(str(checkpoint_file), 1, "500_999bp", 7, 100000)

    loaded = _load_checkpoint(str(checkpoint_file), expected_chunk_size=100000)
    assert loaded == (1, "500_999bp", 7)


def test_checkpoint_load_fails_on_chunk_size_mismatch(tmp_path):
    checkpoint_file = tmp_path / "annotation_checkpoint.txt"

    _save_checkpoint(str(checkpoint_file), 2, "1000_1499bp", 3, 75000)

    with pytest.raises(ValueError, match="chunk_size mismatch"):
        _load_checkpoint(str(checkpoint_file), expected_chunk_size=100000)


def test_legacy_checkpoint_without_chunk_size_still_loads(tmp_path):
    checkpoint_file = tmp_path / "annotation_checkpoint.txt"
    checkpoint_file.write_text("1\t0_499bp\t11\n")

    loaded = _load_checkpoint(str(checkpoint_file), expected_chunk_size=100000)
    assert loaded == (1, "0_499bp", 11)
