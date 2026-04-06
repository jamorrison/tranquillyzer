# __init__.py
"""Utilities for Tranquillyzer."""


def get_version() -> str:
    """Retrieve the tranquillyzer package version."""
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("tranquillyzer")
    except Exception:
        try:
            from tranquillyzer import __version__

            return __version__
        except Exception:
            return "unknown"


def write_tsv_with_version(path, content):
    """Write a TSV/CSV string to a file with a version comment header."""
    with open(path, "w") as fh:
        fh.write(f"# tranquillyzer_version: {get_version()}\n")
        fh.write(content)
