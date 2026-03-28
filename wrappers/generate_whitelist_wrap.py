"""Standalone whitelist generation from existing annotation data.

Discovers cell barcodes via knee-point detection on barcode counts
extracted from annotation outputs, without re-running the annotation pass.
"""

import logging
import os
from collections import Counter

import polars as pl

from scripts.discover_barcodes import (
    _parse_expected_barcode_lengths,
    run_barcode_discovery,
)
from scripts.trained_models import seq_orders

logger = logging.getLogger(__name__)


def _resolve_barcode_columns(barcode_columns_str, model_name, seq_order_file):
    """Resolve barcode column names from explicit string or model config.

    Returns (barcode_columns, seq_order, sequences) where seq_order and
    sequences are None when barcode_columns_str is provided directly.
    """
    if barcode_columns_str:
        barcodes = [c.strip() for c in barcode_columns_str.split(",") if c.strip()]
        if not barcodes:
            raise ValueError("--barcode-columns is empty after parsing.")
        return barcodes, None, None

    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(base_dir, ".."))
    if seq_order_file is None:
        seq_order_file = os.path.join(base_dir, "utils", "seq_orders.yaml")

    seq_order, sequences, barcodes, _umis, _strand = seq_orders(seq_order_file, model_name)
    if not barcodes:
        raise ValueError(
            f"Model '{model_name}' has no barcode columns defined in {seq_order_file}. "
            "Use --barcode-columns to specify them explicitly."
        )
    return barcodes, seq_order, sequences


def _find_annotation_files(output_dir, input_file):
    """Locate annotation data files, returning a list of paths.

    Priority when input_file is None:
      1. chunk files in <output_dir>/annotation_chunks/valid_chunks/
      2. annotations_valid.parquet
      3. annotations_valid_bc_corrected.parquet
    """
    if input_file:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Annotation file not found: {input_file}")
        return [input_file]

    # Check for chunk files
    valid_chunks_dir = os.path.join(output_dir, "annotation_chunks", "valid_chunks")
    if os.path.isdir(valid_chunks_dir):
        chunk_files = sorted(
            os.path.join(valid_chunks_dir, f)
            for f in os.listdir(valid_chunks_dir)
            if f.endswith(".tsv") or f.endswith(".parquet")
        )
        if chunk_files:
            logger.info(f"Found {len(chunk_files)} chunk files in {valid_chunks_dir}")
            return chunk_files

    # Check for combined parquet
    metadata_dir = os.path.join(output_dir, "annotation_metadata")
    for name in ("annotations_valid.parquet", "annotations_valid_bc_corrected.parquet"):
        path = os.path.join(metadata_dir, name)
        if os.path.exists(path):
            logger.info(f"Using combined annotations: {path}")
            return [path]

    raise FileNotFoundError(
        f"No annotation data found. Checked:\n"
        f"  - {valid_chunks_dir}/*.tsv or *.parquet\n"
        f"  - {metadata_dir}/annotations_valid.parquet\n"
        f"  - {metadata_dir}/annotations_valid_bc_corrected.parquet"
    )


def _scan_file(path, needed_cols):
    """Create a polars LazyFrame for a file, selecting only needed columns."""
    if path.endswith(".parquet"):
        lf = pl.scan_parquet(path)
    else:
        lf = pl.scan_csv(path, separator="\t", infer_schema_length=5000)

    # Select only columns that exist in the file
    available = set(lf.collect_schema().names())
    cols_to_select = [c for c in needed_cols if c in available]
    if cols_to_select:
        lf = lf.select(cols_to_select)
    return lf


_INVALID_VALUES = ["", "nan", "none", "NaN", "None", "NAN", "NONE"]


def _rc_expr(col):
    """Reverse-complement a polars string expression (runs in Rust)."""
    return (
        col.str.reverse()
        .str.replace_all("A", "t", literal=True)
        .str.replace_all("T", "a", literal=True)
        .str.replace_all("C", "g", literal=True)
        .str.replace_all("G", "c", literal=True)
        .str.to_uppercase()
    )


def _count_barcodes_from_files(files, barcode_columns, chunk_size):
    """Count barcodes across all annotation files using vectorized polars operations.

    Canonicalizes barcodes (lexicographic min of seq and reverse complement)
    and counts via group_by — all in Rust, no Python per-row loops.
    Returns a Counter of canonicalized barcode tuples.
    """
    seq_cols = [f"{bc}_Sequences" for bc in barcode_columns]
    needed_cols = ["architecture"] + seq_cols
    canon_cols = [f"_canon_{bc}" for bc in barcode_columns]

    frames = []
    for file_path in files:
        lf = _scan_file(file_path, needed_cols)
        lf = lf.filter(pl.col("architecture") == "valid").select(seq_cols)
        for sc, cc in zip(seq_cols, canon_cols):
            col = pl.col(sc).str.strip_chars()
            rc = _rc_expr(col)
            lf = lf.with_columns(pl.min_horizontal(col, rc).alias(cc))
            lf = lf.filter(
                ~pl.col(cc).is_in(_INVALID_VALUES) & pl.col(cc).is_not_null() & (pl.col(cc) != "")
            )
        frames.append(lf.select(canon_cols))

    combined = pl.concat(frames)
    counts_df = combined.group_by(canon_cols).len().collect()

    # Convert to Counter for downstream compatibility
    global_counts = Counter()
    if len(canon_cols) == 1:
        for row in counts_df.iter_rows():
            global_counts[(row[0],)] = row[1]
    else:
        for row in counts_df.iter_rows():
            global_counts[row[:-1]] = row[-1]

    total_reads = sum(global_counts.values())
    logger.info(f"Counted {len(global_counts)} unique barcode tuples from {total_reads} reads across {len(files)} files")
    return global_counts


def generate_whitelist_wrap(
    output_dir,
    model_name="10x3p_sc_ont_011",
    seq_order_file=None,
    input_file=None,
    barcode_columns_str=None,
    expected_cells=None,
    min_cell_ratio=0.50,
    min_reads_per_barcode=3,
    chunk_size=100_000,
):
    """Discover cell barcodes from existing annotation outputs and save a whitelist.

    Parameters
    ----------
    output_dir : str
        Annotation output directory (used for auto-discovery and artifact output).
    model_name : str
        Model name for resolving barcode columns from seq_orders.yaml.
    seq_order_file : str or None
        Path to seq_orders.yaml. Defaults to bundled file.
    input_file : str or None
        Explicit path to annotations file. Bypasses auto-discovery.
    barcode_columns_str : str or None
        Comma-separated barcode column names. Overrides model-based resolution.
    expected_cells : int or None
        Optional hint for knee detection.
    min_cell_ratio : float
        Fraction of cliff-top count used as knee threshold.
    min_reads_per_barcode : int
        Minimum reads for a barcode to be considered.
    chunk_size : int
        Number of rows per streaming chunk.

    Returns
    -------
    str
        Path to the generated discovered_whitelist.tsv.
    """
    # Step 1: Resolve barcode columns
    barcodes, seq_order, sequences = _resolve_barcode_columns(barcode_columns_str, model_name, seq_order_file)
    logger.info(f"Barcode columns: {barcodes}")

    # Step 2: Find annotation data
    annotation_files = _find_annotation_files(output_dir, input_file)

    # Step 3: Count barcodes (column-selective, streamed)
    global_counts = _count_barcodes_from_files(annotation_files, barcodes, chunk_size)

    # Step 4: Parse expected barcode lengths (if model config available)
    expected_bc_lengths = None
    if seq_order is not None and sequences is not None:
        expected_bc_lengths = _parse_expected_barcode_lengths(seq_order, sequences, barcodes)

    # Step 5: Run discovery pipeline
    logger.info("Starting barcode discovery...")
    whitelist_df, _ = run_barcode_discovery(
        global_counts,
        barcodes,
        output_dir,
        expected_cells=expected_cells,
        min_reads=min_reads_per_barcode,
        min_cell_ratio=min_cell_ratio,
        expected_lengths=expected_bc_lengths,
    )

    whitelist_path = os.path.join(output_dir, "annotation_metadata", "discovered_whitelist.tsv")
    logger.info(f"Whitelist generated: {whitelist_path} ({len(whitelist_df)} barcodes)")
    return whitelist_path
