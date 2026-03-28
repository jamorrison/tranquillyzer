import logging

import pytest
import yaml

from scripts.trained_models import (
    _resolve_model,
    get_training_structures,
    get_valid_structures,
    seq_orders,
)


def _write_yaml(tmp_path, config):
    """Write a YAML config to a temp file and return the path string."""
    p = tmp_path / "seq_orders.yaml"
    p.write_text(yaml.dump(config, default_flow_style=False))
    return str(p)


# ── Fixtures ──

@pytest.fixture
def two_level_config():
    """Minimal two-level config for testing."""
    return {
        "libraries": {
            "lib_a": {
                "strand": "fwd",
                "barcodes": ["CBC"],
                "umis": ["UMI"],
                "segments": [
                    {"name": "5p", "pattern": "AAA"},
                    {"name": "CBC", "pattern": "N16"},
                    {"name": "UMI", "pattern": "N10"},
                    {"name": "cDNA", "pattern": "NN"},
                    {"name": "3p", "pattern": "TTT"},
                ],
                "valid_structures": [
                    ["5p", "CBC", "UMI", "cDNA", "3p"],
                ],
                "training_structures": {
                    "full_read": {
                        "order": ["5p", "CBC", "UMI", "cDNA", "3p"],
                        "proportion": 0.70,
                    },
                    "truncated": {
                        "order": ["CBC", "UMI", "cDNA", "3p"],
                        "proportion": 0.30,
                    },
                },
            },
        },
        "models": {
            "model_a": {"library": "lib_a"},
        },
    }


@pytest.fixture
def legacy_config():
    """Old-style flat config for backward compatibility testing."""
    return {
        "model_x": {
            "strand": "rev",
            "barcodes": ["i7"],
            "umis": ["UMI"],
            "segments": [
                {"name": "5p", "pattern": "GGG"},
                {"name": "cDNA", "pattern": "NN"},
            ],
        },
    }


# ── _resolve_model tests ──

class TestResolveModel:
    def test_basic_inheritance(self, two_level_config):
        entry = _resolve_model(two_level_config, "model_a")
        assert entry["strand"] == "fwd"
        assert entry["barcodes"] == ["CBC"]
        assert entry["umis"] == ["UMI"]
        assert len(entry["segments"]) == 5
        assert entry["valid_structures"] == [["5p", "CBC", "UMI", "cDNA", "3p"]]
        assert "full_read" in entry["training_structures"]
        assert "truncated" in entry["training_structures"]

    def test_simple_field_override(self, two_level_config):
        two_level_config["models"]["model_a"]["strand"] = "rev"
        entry = _resolve_model(two_level_config, "model_a")
        assert entry["strand"] == "rev"
        # Other fields still inherited
        assert entry["barcodes"] == ["CBC"]

    def test_training_structures_inherited(self, two_level_config):
        entry = _resolve_model(two_level_config, "model_a")
        ts = entry["training_structures"]
        assert ts["full_read"]["proportion"] == 0.70
        assert ts["truncated"]["proportion"] == 0.30

    def test_training_structures_null_clears(self, two_level_config):
        two_level_config["models"]["model_a"]["training_structures"] = None
        entry = _resolve_model(two_level_config, "model_a")
        assert entry["training_structures"] is None

    def test_training_structures_merge_override(self, two_level_config):
        two_level_config["models"]["model_a"]["training_structures"] = {
            "full_read": {"proportion": 0.80},
        }
        entry = _resolve_model(two_level_config, "model_a")
        ts = entry["training_structures"]
        # Overridden
        assert ts["full_read"]["proportion"] == 0.80
        # Order still inherited from library
        assert ts["full_read"]["order"] == ["5p", "CBC", "UMI", "cDNA", "3p"]
        # Other entry still present
        assert ts["truncated"]["proportion"] == 0.30

    def test_training_structures_merge_add(self, two_level_config):
        two_level_config["models"]["model_a"]["training_structures"] = {
            "new_struct": {
                "order": ["5p", "cDNA", "3p"],
                "proportion": 0.10,
            },
        }
        entry = _resolve_model(two_level_config, "model_a")
        ts = entry["training_structures"]
        assert "full_read" in ts
        assert "truncated" in ts
        assert "new_struct" in ts
        assert ts["new_struct"]["proportion"] == 0.10

    def test_training_structures_merge_remove(self, two_level_config):
        two_level_config["models"]["model_a"]["training_structures"] = {
            "truncated": None,
        }
        entry = _resolve_model(two_level_config, "model_a")
        ts = entry["training_structures"]
        assert "full_read" in ts
        assert "truncated" not in ts

    def test_model_missing_raises(self, two_level_config):
        with pytest.raises(ValueError, match="not_a_model"):
            _resolve_model(two_level_config, "not_a_model")

    def test_library_missing_raises(self, two_level_config):
        two_level_config["models"]["bad"] = {"library": "no_such_lib"}
        with pytest.raises(ValueError, match="no_such_lib"):
            _resolve_model(two_level_config, "bad")

    def test_no_library_key_raises(self, two_level_config):
        two_level_config["models"]["bad"] = {"strand": "fwd"}
        with pytest.raises(ValueError, match="must have a 'library' field"):
            _resolve_model(two_level_config, "bad")

    def test_legacy_format(self, legacy_config):
        entry = _resolve_model(legacy_config, "model_x")
        assert entry["strand"] == "rev"
        assert entry["barcodes"] == ["i7"]

    def test_legacy_format_missing_raises(self, legacy_config):
        with pytest.raises(ValueError, match="nope"):
            _resolve_model(legacy_config, "nope")

    def test_deep_copy_isolation(self, two_level_config):
        """Resolving a model should not mutate the original config."""
        entry1 = _resolve_model(two_level_config, "model_a")
        entry1["strand"] = "MUTATED"
        entry2 = _resolve_model(two_level_config, "model_a")
        assert entry2["strand"] == "fwd"


# ── Public function tests with new format ──

class TestPublicFunctionsNewFormat:
    def test_seq_orders(self, tmp_path, two_level_config):
        path = _write_yaml(tmp_path, two_level_config)
        seq_order, sequences, barcodes, umis, strand = seq_orders(path, "model_a")
        assert seq_order == ["5p", "CBC", "UMI", "cDNA", "3p"]
        assert sequences == ["AAA", "N16", "N10", "NN", "TTT"]
        assert barcodes == ["CBC"]
        assert umis == ["UMI"]
        assert strand == "fwd"

    def test_get_valid_structures(self, tmp_path, two_level_config):
        path = _write_yaml(tmp_path, two_level_config)
        vs = get_valid_structures(path, "model_a")
        assert vs == [["5p", "CBC", "UMI", "cDNA", "3p"]]

    def test_get_valid_structures_default(self, tmp_path, two_level_config):
        del two_level_config["libraries"]["lib_a"]["valid_structures"]
        path = _write_yaml(tmp_path, two_level_config)
        vs = get_valid_structures(path, "model_a")
        assert vs == [["5p", "CBC", "UMI", "cDNA", "3p"]]

    def test_get_training_structures(self, tmp_path, two_level_config):
        path = _write_yaml(tmp_path, two_level_config)
        ts = get_training_structures(path, "model_a")
        assert isinstance(ts, list)
        assert len(ts) == 2
        names = {t["order"][0] for t in ts}
        # full_read starts with 5p, truncated starts with CBC
        assert "5p" in names
        assert "CBC" in names
        for t in ts:
            assert "patterns" in t
            assert "repeat" in t
            assert "proportion" in t
            assert "rc_elements" in t
            assert "rc_pattern" in t

    def test_get_training_structures_null_fallback(self, tmp_path, two_level_config):
        two_level_config["models"]["model_a"]["training_structures"] = None
        path = _write_yaml(tmp_path, two_level_config)
        ts = get_training_structures(path, "model_a")
        assert len(ts) == 1
        assert ts[0]["proportion"] == 1.0
        assert ts[0]["order"] == ["5p", "CBC", "UMI", "cDNA", "3p"]

    def test_proportion_warning(self, tmp_path, two_level_config, caplog):
        two_level_config["libraries"]["lib_a"]["training_structures"]["full_read"]["proportion"] = 0.10
        # total = 0.10 + 0.30 = 0.40
        path = _write_yaml(tmp_path, two_level_config)
        with caplog.at_level(logging.WARNING):
            get_training_structures(path, "model_a")
        assert "sum to 0.4000" in caplog.text


# ── Legacy format backward compatibility ──

class TestLegacyBackwardCompat:
    def test_seq_orders_legacy(self, tmp_path, legacy_config):
        path = _write_yaml(tmp_path, legacy_config)
        seq_order, sequences, barcodes, umis, strand = seq_orders(path, "model_x")
        assert seq_order == ["5p", "cDNA"]
        assert strand == "rev"

    def test_get_valid_structures_legacy(self, tmp_path, legacy_config):
        path = _write_yaml(tmp_path, legacy_config)
        vs = get_valid_structures(path, "model_x")
        assert vs == [["5p", "cDNA"]]

    def test_get_training_structures_legacy(self, tmp_path, legacy_config):
        path = _write_yaml(tmp_path, legacy_config)
        ts = get_training_structures(path, "model_x")
        assert len(ts) == 1
        assert ts[0]["proportion"] == 1.0
