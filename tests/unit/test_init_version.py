"""Tests for __init__.py and _version.py version metadata."""

import sys
import types
from unittest.mock import patch


def test_version_fallback_to_version_module():
    """When importlib.metadata.version fails, fall back to _version module."""
    from __init__ import _resolve_version

    fake_version = "1.2.3.test"
    fake_mod = types.ModuleType("_version")
    fake_mod.__version__ = fake_version

    with patch("importlib.metadata.version", side_effect=Exception("not installed")):
        sys.modules["_version"] = fake_mod
        try:
            assert _resolve_version() == fake_version
        finally:
            sys.modules.pop("_version", None)


def test_version_fallback_to_default():
    """When both importlib.metadata.version and _version fail, default to 0.0.0."""
    from __init__ import _resolve_version

    sys.modules.pop("_version", None)

    with patch("importlib.metadata.version", side_effect=Exception("not installed")):
        with patch.dict(sys.modules, {"_version": None}):
            assert _resolve_version() == "0.0.0"


def test_version_module_exports():
    """Verify _version.py exposes expected version metadata."""
    import _version

    assert isinstance(_version.__version__, str)
    assert isinstance(_version.version, str)
    assert _version.__version__ == _version.version

    assert isinstance(_version.__version_tuple__, tuple)
    assert _version.__version_tuple__ == _version.version_tuple

    assert _version.__commit_id__ == _version.commit_id

    assert set(_version.__all__) == {
        "__version__",
        "__version_tuple__",
        "version",
        "version_tuple",
        "__commit_id__",
        "commit_id",
    }
