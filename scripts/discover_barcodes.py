"""Whitelist-free barcode discovery from annotation data.

Discovers true cell barcodes via count-based knee-point detection,
merges near-duplicate sequences, and builds a synthetic whitelist
compatible with the existing barcode correction pipeline.
"""

import json
import logging
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from scripts.correct_barcodes import reverse_complement

logger = logging.getLogger(__name__)


def _parse_expected_barcode_lengths(seq_order, sequences, barcode_columns):
    """Extract expected barcode lengths from seq_order segment patterns.

    Returns a dict mapping barcode column name to expected length (int),
    or None if the pattern is variable-length (NN).
    """
    segment_patterns = dict(zip(seq_order, sequences))
    expected = {}
    for bc in barcode_columns:
        pattern = segment_patterns.get(bc)
        if pattern:
            match = re.fullmatch(r"N(\d+)", pattern)
            expected[bc] = int(match.group(1)) if match else None
        else:
            expected[bc] = None
    return expected


def canonicalize_barcode(seq):
    """Return the canonical form of a barcode (lexicographic min of seq and its reverse complement)."""
    return min(seq, reverse_complement(seq))


def _parse_barcode_seq(value):
    """Return a cleaned barcode sequence string, or None if missing/invalid."""
    if not value or not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped.upper() in {"", "NAN", "NONE"}:
        return None
    return stripped


def count_barcodes_in_chunk(chunk_df, barcode_columns):
    """Count observed barcode tuples in an annotation chunk DataFrame.

    For single-barcode libraries, counts individual barcodes.
    For multi-barcode libraries, counts co-occurring barcode combinations
    (tuples) from the same read — only reads with all barcodes present are counted.

    Uses vectorized pandas operations for performance on large chunks.

    Parameters
    ----------
    chunk_df : pd.DataFrame
        Chunk of annotation results. Must contain ``{bc}_Sequences`` columns
        and an ``architecture`` column (only ``"valid"`` rows are counted).
    barcode_columns : list[str]
        Barcode column names (e.g. ``["CBC"]`` or ``["CBC1", "CBC2"]``).

    Returns
    -------
    Counter
        Counter of canonicalized barcode tuples. Keys are tuples of
        canonicalized sequences in the same order as ``barcode_columns``.
        For single-barcode libraries, keys are 1-tuples.
    """
    if "architecture" not in chunk_df.columns:
        return Counter()

    # Vectorized: filter to valid rows
    valid = chunk_df[chunk_df["architecture"] == "valid"]
    if valid.empty:
        return Counter()

    # Check all required sequence columns exist
    seq_cols = [f"{bc}_Sequences" for bc in barcode_columns]
    for col in seq_cols:
        if col not in valid.columns:
            return Counter()

    # Vectorized: extract and canonicalize barcode sequences
    canon_series = []
    for col in seq_cols:
        s = valid[col].astype(str).str.strip()
        # Mask out invalid values
        invalid_mask = s.isin({"", "nan", "None", "NaN", "none", "NAN", "NONE"}) | s.isna()
        s = s.where(~invalid_mask)
        s = s.map(canonicalize_barcode, na_action="ignore")
        canon_series.append(s)

    # Drop rows where any barcode is missing
    combined = pd.concat(canon_series, axis=1)
    combined = combined.dropna()
    if combined.empty:
        return Counter()

    # Build tuples and count
    if len(barcode_columns) == 1:
        # Fast path: single barcode — avoid tuple overhead
        value_counts = combined.iloc[:, 0].value_counts()
        return Counter({(k,): v for k, v in value_counts.items()})
    else:
        # Multi-barcode: zip columns into tuples
        tuples = list(zip(*(combined.iloc[:, i] for i in range(len(barcode_columns)))))
        return Counter(tuples)


def detect_knee_point(counts, expected_cells=None, min_reads=3, min_cell_ratio=0.01):
    """Identify true cell barcodes using quantile-based thresholding.

    Uses a wf-single-cell-inspired approach: rank barcodes by count, estimate
    the cell population size, then set a threshold at p95/20 of the top N counts.

    Parameters
    ----------
    counts : Counter
        Barcode -> read count mapping.
    expected_cells : int or None
        If provided, use as the cell count estimate directly.
        If None, auto-estimate by finding where counts drop below
        ``top_count * min_cell_ratio``.
    min_reads : int
        Minimum reads to consider a barcode (pre-filter).
    min_cell_ratio : float
        In auto mode: fraction of the top barcode's count used as the
        drop threshold for estimating cell count (default 0.01 = 1%).

    Returns
    -------
    tuple[list, float]
        (barcodes identified as true cells sorted by count descending, threshold value).
    """
    if not counts:
        return [], 0.0

    # Sort by count descending
    sorted_barcodes = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # Pre-filter by minimum reads
    sorted_barcodes = [(bc, c) for bc, c in sorted_barcodes if c >= min_reads]
    if not sorted_barcodes:
        return [], 0.0

    counts_array = np.array([c for _, c in sorted_barcodes], dtype=float)

    if expected_cells is not None and expected_cells > 0:
        # Guided mode: use the count at the expected_cells rank as the cliff-top reference
        cliff_top = min(expected_cells, len(counts_array)) - 1
        cliff_top_count = counts_array[cliff_top]
        logger.info(f"Guided mode: using expected_cells={expected_cells}, cliff_top_count={cliff_top_count:.0f}")
    else:
        # Auto mode: kneedle + percentile threshold
        # Step 1: kneedle (convex) on log-log curve to find cliff top
        log_ranks = np.log10(np.arange(1, len(counts_array) + 1))
        log_counts = np.log10(counts_array)

        x_norm = (log_ranks - log_ranks[0]) / (log_ranks[-1] - log_ranks[0])
        y_norm = (log_counts - log_counts[-1]) / (log_counts[0] - log_counts[-1])

        distances = y_norm + x_norm - 1
        cliff_top = int(np.argmax(distances))
        cliff_top_count = counts_array[cliff_top]

        logger.info(f"Kneedle cliff top: rank {cliff_top + 1}, count={cliff_top_count:.0f}")

    # Step 2: threshold = fraction of cliff-top count
    threshold = max(min_reads, cliff_top_count * min_cell_ratio)
    discovered = [bc for bc, c in sorted_barcodes if c >= threshold]
    logger.info(
        f"Knee detection: cliff_top_count={cliff_top_count:.0f}, "
        f"min_cell_ratio={min_cell_ratio}, threshold={threshold:.1f}, "
        f"discovered={len(discovered)} barcodes"
    )

    if len(discovered) > 50000:
        logger.warning(
            f"Large number of discovered barcodes ({len(discovered)}). "
            f"Consider providing --expected-cells or adjusting --min-cell-ratio."
        )

    return discovered, threshold


def _tuple_distance(a, b):
    """Compute total Levenshtein distance between two barcode tuples (sum of per-component distances)."""
    return sum(Levenshtein.distance(x, y) for x, y in zip(a, b))


def _deletion_keys_for_tuple(bc_tuple):
    """Generate deletion neighborhood keys for a barcode tuple.

    For max_dist=1, two tuples that differ by at most 1 edit (substitution/indel)
    in total across all components will share at least one deletion key.

    For each component, generate keys by deleting one character at a time
    while keeping all other components intact.
    """
    keys = []
    for comp_idx, component in enumerate(bc_tuple):
        for char_idx in range(len(component)):
            deleted = component[:char_idx] + component[char_idx + 1 :]
            key = bc_tuple[:comp_idx] + (deleted,) + bc_tuple[comp_idx + 1 :]
            keys.append(key)
    return keys


def merge_near_duplicate_barcodes(barcodes, counts, max_dist=1):
    """Merge near-duplicate barcode tuples using greedy highest-count strategy.

    Uses deletion neighborhood hashing for O(K * L) lookups instead of
    O(K²) brute-force comparisons. For each barcode tuple, generates deletion
    keys (one character removed per position per component). Two tuples within
    edit distance 1 share at least one deletion key.

    Parameters
    ----------
    barcodes : list[tuple[str, ...]]
        Discovered barcode tuples.
    counts : Counter
        Barcode tuple -> read count mapping.
    max_dist : int
        Maximum total Levenshtein distance for merging (default 1).

    Returns
    -------
    dict[tuple, tuple]
        Mapping from variant tuple -> canonical tuple.
        Canonical tuples map to themselves.
    """
    from collections import defaultdict

    # Sort by count descending — high-count barcodes become canonical first
    sorted_bcs = sorted(barcodes, key=lambda bc: counts.get(bc, 0), reverse=True)

    logger.info(f"Near-duplicate merging: processing {len(sorted_bcs)} barcodes with deletion neighborhood index")

    # Build deletion index: maps deletion keys to canonical barcodes
    deletion_index = defaultdict(list)

    canonical_set = set()
    mapping = {}

    for bc in sorted_bcs:
        if bc in mapping:
            continue

        # Look up this barcode's deletion keys in the index to find canonical neighbors
        found_canon = None
        del_keys = _deletion_keys_for_tuple(bc)
        for key in del_keys:
            for candidate in deletion_index.get(key, []):
                if _tuple_distance(bc, candidate) <= max_dist:
                    found_canon = candidate
                    break
            if found_canon:
                break

        # Check if this barcode itself is a deletion key of an existing canonical
        # (handles shorter variants that are deletions of longer canonicals)
        if not found_canon:
            for candidate in deletion_index.get(bc, []):
                if candidate != bc and _tuple_distance(bc, candidate) <= max_dist:
                    found_canon = candidate
                    break

        # Also check exact match (distance 0)
        if not found_canon and bc in canonical_set:
            found_canon = bc

        if found_canon and found_canon != bc:
            mapping[bc] = found_canon
            counts[found_canon] += counts.get(bc, 0)
        else:
            # This barcode becomes canonical — add its deletion keys to the index
            canonical_set.add(bc)
            mapping[bc] = bc
            for key in del_keys:
                deletion_index[key].append(bc)
            # Also index the identity key for exact matches
            deletion_index[bc].append(bc)

    n_merged = sum(1 for k, v in mapping.items() if k != v)
    logger.info(f"Near-duplicate merging: {n_merged} barcode tuples merged into {len(canonical_set)} canonical entries")
    return mapping


def build_whitelist_from_discovery(canonical_tuples, barcode_columns):
    """Build whitelist DataFrame and dict from discovered barcode tuples.

    Parameters
    ----------
    canonical_tuples : list[tuple[str, ...]]
        List of canonical barcode tuples. Each tuple has one element per
        barcode column, in the same order as ``barcode_columns``.
    barcode_columns : list[str]
        Barcode column names.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(whitelist_df, whitelist_dict)`` compatible with ``bc_n_demultiplex``.
    """
    # Build DataFrame: each row is a discovered barcode combination
    data = {bc_col: [t[i] for t in canonical_tuples] for i, bc_col in enumerate(barcode_columns)}
    whitelist_df = pd.DataFrame(data)

    # Build whitelist_dict in the same format the correction pipeline expects
    whitelist_dict = {
        "cell_ids": {
            idx + 1: "-".join(str(v) for v in row.dropna().unique())
            for idx, row in whitelist_df[barcode_columns].iterrows()
        },
        **{bc_col: whitelist_df[bc_col].dropna().unique().tolist() for bc_col in barcode_columns},
    }

    return whitelist_df, whitelist_dict


def save_discovered_whitelist(whitelist_df, output_path):
    """Write discovered whitelist to TSV file.

    Parameters
    ----------
    whitelist_df : pd.DataFrame
        Whitelist DataFrame from ``build_whitelist_from_discovery``.
    output_path : str
        Path to write the TSV file.
    """
    from utils import write_tsv_with_version

    write_tsv_with_version(output_path, whitelist_df.to_csv(sep="\t", index=False))
    logger.info(f"Saved discovered whitelist ({len(whitelist_df)} barcodes) to {output_path}")


def save_discovery_stats(output_path, barcode_columns, tuple_counts, canonical_tuples, merged_mapping):
    """Write barcode discovery summary statistics to JSON.

    Parameters
    ----------
    output_path : str
        Path to write the JSON file.
    barcode_columns : list[str]
        Barcode column names.
    tuple_counts : Counter
        Raw barcode tuple counts.
    canonical_tuples : list[tuple[str, ...]]
        Discovered canonical barcode tuples.
    merged_mapping : dict[tuple, tuple]
        Variant-to-canonical tuple mappings.
    """
    n_canonical = len(set(merged_mapping.values())) if merged_mapping else len(canonical_tuples)
    from utils import get_version

    stats = {
        "tranquillyzer_version": get_version(),
        "barcode_columns": barcode_columns,
        "unique_barcode_tuples_observed": len(tuple_counts),
        "total_reads_with_barcodes": sum(tuple_counts.values()),
        "tuples_above_knee": len(canonical_tuples) + sum(1 for k, v in merged_mapping.items() if k != v),
        "canonical_cells_after_merge": n_canonical,
    }
    # Per-column stats
    for i, bc_col in enumerate(barcode_columns):
        unique_seqs = {t[i] for t in tuple_counts}
        canonical_seqs = {t[i] for t in canonical_tuples}
        stats[f"{bc_col}_unique_observed"] = len(unique_seqs)
        stats[f"{bc_col}_canonical"] = len(canonical_seqs)

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved discovery stats to {output_path}")


def plot_barcode_rank(counts, n_knee, output_path):
    """Save a log-log rank plot of barcode counts with vertical knee line.

    Parameters
    ----------
    counts : Counter
        Barcode tuple -> read count mapping.
    n_knee : int
        Number of barcodes above the knee threshold (vertical line position).
    output_path : str
        Path to save the PNG file.
    """
    sorted_counts = sorted(counts.values(), reverse=True)
    if not sorted_counts:
        return

    ranks = list(range(1, len(sorted_counts) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ranks, sorted_counts, color="#4C78A8", linewidth=1.5)
    ax.axvline(x=n_knee, color="#E45756", linewidth=1, linestyle="--", label=f"Knee ({n_knee:,} barcodes)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Barcode rank")
    ax.set_ylabel("Read count")
    ax.set_title("Barcode Rank Plot (whitelist-free discovery)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    from utils import get_version

    fig.savefig(output_path, dpi=150, metadata={"Software": f"tranquillyzer v{get_version()}"})
    plt.close(fig)
    logger.info(f"Saved barcode rank plot to {output_path}")


def save_barcode_counts_tsv(counts, barcode_columns, knee_tuples, mapping, output_path):
    """Write all observed barcode tuples with counts and discovery status to TSV.

    Parameters
    ----------
    counts : Counter
        Barcode tuple -> read count mapping (raw, pre-merge).
    barcode_columns : list[str]
        Barcode column names.
    knee_tuples : list[tuple]
        Barcode tuples that passed knee detection.
    mapping : dict[tuple, tuple]
        Variant -> canonical tuple mapping from near-duplicate merging.
    output_path : str
        Path to write the TSV file.
    """
    knee_set = set(knee_tuples)
    canonical_set = set(mapping.values()) if mapping else set()

    rows = []
    for bc_tuple, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        row = {}
        for i, col in enumerate(barcode_columns):
            row[col] = bc_tuple[i]
        row["read_count"] = count
        if bc_tuple in canonical_set:
            row["status"] = "canonical"
        elif bc_tuple in knee_set:
            canon = mapping.get(bc_tuple, bc_tuple)
            canon_str = "-".join(canon)
            row["status"] = f"merged_into:{canon_str}"
        else:
            row["status"] = "below_knee"
        rows.append(row)

    df = pd.DataFrame(rows)
    from utils import write_tsv_with_version

    write_tsv_with_version(output_path, df.to_csv(sep="\t", index=False))
    logger.info(f"Saved barcode counts ({len(df)} entries) to {output_path}")


def run_barcode_discovery(
    global_counts,
    barcode_columns,
    output_dir,
    expected_cells=None,
    min_reads=3,
    min_cell_ratio=0.01,
    merge_dist=1,
    expected_lengths=None,
):
    """Run the full barcode discovery pipeline: knee detection, near-duplicate merging, whitelist building.

    Operates on barcode tuples (combinations) rather than individual barcode columns.
    For single-barcode libraries, tuples are 1-element. For multi-barcode libraries,
    tuples represent co-occurring barcode combinations from the same read.

    Parameters
    ----------
    global_counts : Counter
        Merged barcode tuple counts from all annotation chunks.
        Keys are tuples of canonicalized barcode sequences.
    barcode_columns : list[str]
        Barcode column names.
    output_dir : str
        Output directory for whitelist and stats files.
    expected_cells : int or None
        Optional hint for knee detection.
    min_reads : int
        Minimum reads to consider a barcode tuple.
    min_cell_ratio : float
        Threshold ratio for guided knee detection.
    merge_dist : int
        Maximum total Levenshtein distance for near-duplicate merging.
    expected_lengths : dict or None
        Mapping of barcode column name to expected sequence length (int).
        Barcodes not matching their expected length are filtered out before
        knee detection. ``None`` values skip filtering for that column.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(whitelist_df, whitelist_dict)`` for use with ``bc_n_demultiplex``.
    """
    metadata_dir = os.path.join(output_dir, "annotation_metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    if not global_counts:
        logger.warning("No barcode counts available. Returning empty whitelist.")
        whitelist_df = pd.DataFrame(columns=barcode_columns)
        whitelist_dict = {"cell_ids": {}, **{bc: [] for bc in barcode_columns}}
        return whitelist_df, whitelist_dict

    logger.info(f"Discovering barcodes: {len(global_counts)} unique tuples across {len(barcode_columns)} columns")

    # Step 0: Filter by expected barcode length
    if expected_lengths:
        pre_filter_count = len(global_counts)
        filtered = Counter()
        for bc_tuple, count in global_counts.items():
            if all(
                expected_lengths.get(col) is None or len(bc_tuple[i]) == expected_lengths[col]
                for i, col in enumerate(barcode_columns)
            ):
                filtered[bc_tuple] = count
        n_removed = pre_filter_count - len(filtered)
        if n_removed:
            logger.info(
                f"Length filter: removed {n_removed} barcode tuples with unexpected lengths "
                f"(expected: {expected_lengths})"
            )
        global_counts = filtered

    # Step 1: Knee-point detection on tuple counts
    knee_tuples, threshold = detect_knee_point(global_counts, expected_cells, min_reads, min_cell_ratio)

    # Step 2: Rank plot (before merge — shows raw count distribution)
    plot_barcode_rank(global_counts, len(knee_tuples), os.path.join(metadata_dir, "barcode_rank_plot.png"))

    # Step 3: Near-duplicate merging
    mapping = merge_near_duplicate_barcodes(knee_tuples, global_counts, max_dist=merge_dist)
    canonical_tuples = sorted(set(mapping.values()), key=lambda t: global_counts.get(t, 0), reverse=True)

    # Step 4: Build whitelist
    whitelist_df, whitelist_dict = build_whitelist_from_discovery(canonical_tuples, barcode_columns)

    # Save artifacts
    save_discovered_whitelist(whitelist_df, os.path.join(metadata_dir, "discovered_whitelist.tsv"))
    save_discovery_stats(
        os.path.join(metadata_dir, "barcode_discovery_stats.json"),
        barcode_columns,
        global_counts,
        canonical_tuples,
        mapping,
    )
    save_barcode_counts_tsv(
        global_counts,
        barcode_columns,
        knee_tuples,
        mapping,
        os.path.join(metadata_dir, "barcode_counts.tsv"),
    )

    return whitelist_df, whitelist_dict
