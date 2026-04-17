import gc
import logging
import multiprocessing as mp
import numpy as np
import tensorflow as tf
from rapidfuzz.distance import Levenshtein
from scripts.correct_barcodes import reverse_complement

logger = logging.getLogger(__name__)

_POLY_SET = {"polyA", "polyT"}


def _labels_match(a, b):
    """Match two labels, treating polyA and polyT as interchangeable."""
    return a == b or (a in _POLY_SET and b in _POLY_SET)


# ======================= collapse labels into order ======================= #


def collapse_labels(arr, read_length):
    """Collapse consecutive identical per-position labels into (label, start, end) spans."""
    read = arr[0:read_length]
    collapsed_array = []
    count_dict = {}
    indices_dict = {}
    prev = None
    start_index = 0

    for i, element in enumerate(read):
        if element != prev:
            if prev is not None:
                collapsed_array.append(prev)
                count_dict[prev] = count_dict.get(prev, 0) + 1
                indices_dict[prev] = indices_dict.get(prev, []) + [(start_index, i)]
            prev = element
            start_index = i

    if prev is not None:
        collapsed_array.append(prev)
        count_dict[prev] = count_dict.get(prev, 0) + 1
        indices_dict[prev] = indices_dict.get(prev, []) + [(start_index, len(read))]

    return collapsed_array, count_dict, indices_dict


# ============ check if elements are in expected order ============ #


def flexible_sliding_match(array, pattern):
    """
    Find full pattern matches allowing 'cDNA' at the start, end, and between concatenated patterns.
    Matches must be exact — 'cDNA' is NOT allowed within a single match.
    """
    # Trim leading and trailing cDNA
    start = 0
    while start < len(array) and array[start] == "cDNA":
        start += 1
    end = len(array)
    while end > start and array[end - 1] == "cDNA":
        end -= 1

    core_array = array[start:end]

    matches = []
    i = 0
    while i <= len(core_array) - len(pattern):
        window = core_array[i : i + len(pattern)]
        if all(_labels_match(w, p) for w, p in zip(window, pattern)):
            # Match found
            matches.append((start + i, start + i + len(pattern) - 1))
            i += len(pattern)
            # Skip trailing cDNA if present before next pattern
            while i < len(core_array) and core_array[i] == "cDNA":
                i += 1
        else:
            i += 1

    return matches


def check_order(collapsed_array, count_dict, expected_orders):
    """Check if collapsed_array matches ANY of the expected orders.

    Uses unified multi-pattern matching so different fragments in a
    concatenated read can match different valid structures.

    Args:
        expected_orders: list of acceptable segment orders (list of lists).
    """
    # 1. Generate all candidate matches from all patterns/orientations
    candidates = []
    for expected_order in expected_orders:
        for orientation, pattern in [("+", expected_order), ("-", expected_order[::-1])]:
            matches = flexible_sliding_match(collapsed_array, pattern)
            for match_start, match_end in matches:
                candidates.append((match_start, match_end, orientation, len(pattern)))

    if not candidates:
        reason = "Unexpected pattern: [" + "_".join(collapsed_array) + "]"
        return False, "", reason, []

    # 2. Sort: longer patterns first (more specific), then by start position
    candidates.sort(key=lambda x: (-x[3], x[0]))

    # 3. Greedy non-overlapping selection (boundary-only overlap allowed)
    matched_positions = set()
    match_details = []
    all_orientations = {}
    first_orientation = ""

    for match_start, match_end, orientation, pat_len in candidates:
        region = set(range(match_start, match_end + 1))
        overlap = region & matched_positions
        if overlap and not overlap <= {match_start, match_end}:
            continue
        matched_positions.update(region)
        if not first_orientation:
            first_orientation = orientation
        all_orientations[orientation] = all_orientations.get(orientation, 0) + 1
        match_details.append((match_start, match_end, orientation))

    # 4. Sort match_details by position for consistent downstream processing
    match_details.sort(key=lambda x: x[0])

    # 5. Build result
    total = sum(all_orientations.values())
    orientation = first_orientation
    breakdown = ", ".join(f"{k}:{v}" for k, v in all_orientations.items())

    unmatched = [label for i, label in enumerate(collapsed_array) if i not in matched_positions and label != "cDNA"]
    extra_info = f" — extra segments: [{'_'.join(unmatched)}]" if unmatched else ""

    if total == 1 and not unmatched:
        return True, orientation, "valid", match_details
    else:
        reason = f"concatenated reads x{total} ({breakdown}){extra_info}"
        return False, orientation, reason, match_details


# =================== split concatenated reads =================== #


def _collapsed_to_base_coords(collapsed_array, indices_dict):
    """Map each collapsed_array index to its (base_start, base_end) coordinates."""
    label_occ = {}
    coords = []
    for label in collapsed_array:
        occ = label_occ.get(label, 0)
        coords.append(indices_dict[label][occ])
        label_occ[label] = occ + 1
    return coords


def _build_fragment_annotation(
    read,
    read_length,
    collapsed_array,
    indices_dict,
    match_start,
    match_end,
    orientation,
    seq_order,
    barcodes,
    frag_idx,
    total_frags,
    known_patterns=None,
):
    """Build an annotation dict for a single fragment extracted from a concatenated read."""
    base_coords = _collapsed_to_base_coords(collapsed_array, indices_dict)

    annotations = {element: {"Starts": [], "Ends": [], "Sequences": []} for element in seq_order}
    annotations["random_s"] = {"Starts": [], "Ends": [], "Sequences": []}
    annotations["random_e"] = {"Starts": [], "Ends": [], "Sequences": []}
    annotations["read"] = read[0:read_length]

    for j in range(match_start, match_end + 1):
        label = collapsed_array[j]
        if label in annotations and isinstance(annotations[label], dict):
            base_start, base_end = base_coords[j]
            annotations[label]["Starts"].append(base_start)
            annotations[label]["Ends"].append(base_end)

    annotations["architecture"] = "valid"
    annotations["read_length"] = str(read_length)
    annotations["orientation"] = orientation
    annotations["reason"] = f"split from concatenated (frag {frag_idx}/{total_frags})"

    for barcode in barcodes:
        if annotations[barcode]["Starts"]:
            annotations[barcode]["Sequences"] = [
                read[int(annotations[barcode]["Starts"][0]) : int(annotations[barcode]["Ends"][0])]
            ]

    if known_patterns:
        for seg, pattern in known_patterns.items():
            if seg in annotations and annotations[seg]["Starts"]:
                rc_pattern = reverse_complement(pattern)
                seqs = []
                dists = []
                for s, e in zip(annotations[seg]["Starts"], annotations[seg]["Ends"]):
                    seg_seq = read[int(s) : int(e)]
                    seqs.append(seg_seq)
                    dist_fwd = Levenshtein.distance(seg_seq, pattern)
                    dist_rc = Levenshtein.distance(seg_seq, rc_pattern)
                    dists.append(min(dist_fwd, dist_rc))
                annotations[seg]["Sequences"] = seqs
                annotations[seg]["EditDist"] = dists

    return annotations


# =================== process full-length reads =================== #


def process_full_len_reads(data, barcodes, label_binarizer, model_path, split_concatenated=False, known_patterns=None):
    """Process a batch of reads: collapse labels, validate structure, extract annotations."""
    read, prediction, read_length, seq_order, valid_structs = data

    prediction = np.asarray(prediction)
    if prediction.ndim == 1:
        prediction = prediction[np.newaxis, :]
    decoded_prediction = label_binarizer.classes_[prediction[0] if prediction.ndim == 2 else prediction]

    # read_length = read_length - 1

    decoded_prediction = decoded_prediction[0:read_length]

    collapsed_array, count_dict, indices_dict = collapse_labels(decoded_prediction, read_length)

    # Normalize polyA/polyT: if the model predicted the opposite of what seq_order uses,
    # merge those coordinates under the canonical name so annotations are populated correctly.
    _poly_canonical = next((x for x in seq_order if x in _POLY_SET), None)
    if _poly_canonical is not None:
        _poly_other = "polyT" if _poly_canonical == "polyA" else "polyA"
        if _poly_other in indices_dict:
            indices_dict.setdefault(_poly_canonical, []).extend(indices_dict.pop(_poly_other))
        # Also normalize collapsed_array so check_order sees the canonical label
        collapsed_array = [_poly_canonical if x in _POLY_SET else x for x in collapsed_array]
        # Keep indices_dict sorted by position after merging poly variants
        if _poly_canonical in indices_dict:
            indices_dict[_poly_canonical].sort()

    order_match, order, reasons, match_details = check_order(collapsed_array, count_dict, valid_structs)

    # --- Split concatenated reads into individual fragments ---
    if not order_match and split_concatenated and match_details:
        total_frags = len(match_details)
        results = []

        # 1) Extract each full fragment as a valid annotation
        for frag_idx, (m_start, m_end, m_orient) in enumerate(match_details, 1):
            frag_ann = _build_fragment_annotation(
                read,
                read_length,
                collapsed_array,
                indices_dict,
                m_start,
                m_end,
                m_orient,
                seq_order,
                barcodes,
                frag_idx,
                total_frags,
                known_patterns,
            )
            frag_ann["reason"] = (
                f"split from concatenated (frag {frag_idx}/{total_frags}) — parent read in invalid annotations"
            )
            results.append(frag_ann)

        # 2) Emit the whole read as an invalid remainder so it is not lost
        remainder = {element: {"Starts": [], "Ends": [], "Sequences": []} for element in seq_order}
        remainder["random_s"] = {"Starts": [], "Ends": [], "Sequences": []}
        remainder["random_e"] = {"Starts": [], "Ends": [], "Sequences": []}
        remainder["read"] = read[0:read_length]
        for element in indices_dict:
            for coordinates in indices_dict[element]:
                start, end = coordinates
                remainder[element]["Starts"].append(start)
                remainder[element]["Ends"].append(end)
        remainder["architecture"] = "invalid"
        remainder["read_length"] = str(read_length)
        remainder["orientation"] = order
        remainder["reason"] = f"{reasons} — {total_frags} fragment(s) extracted to valid annotations"
        results.append(remainder)

        return results

    annotations = {element: {"Starts": [], "Ends": [], "Sequences": []} for element in seq_order}
    annotations["random_s"] = {"Starts": [], "Ends": [], "Sequences": []}
    annotations["random_e"] = {"Starts": [], "Ends": [], "Sequences": []}
    annotations["read"] = read[0:read_length]

    for element in indices_dict:
        for coordinates in indices_dict[element]:
            start, end = coordinates
            annotations[element]["Starts"].append(start)
            annotations[element]["Ends"].append(end)
    if order_match:
        if len(annotations["cDNA"]["Starts"]) == 2:
            if order == "+" and collapsed_array[0] == "cDNA":
                annotations["random_s"]["Starts"].append(annotations["cDNA"]["Starts"][0])
                annotations["random_s"]["Ends"].append(annotations["cDNA"]["Ends"][0])
                annotations["cDNA"]["Starts"] = [annotations["cDNA"]["Starts"][1]]
                annotations["cDNA"]["Ends"] = [annotations["cDNA"]["Ends"][1]]
            elif order == "+" and collapsed_array[-1] == "cDNA":
                annotations["random_e"]["Starts"].append(annotations["cDNA"]["Starts"][1])
                annotations["random_e"]["Ends"].append(annotations["cDNA"]["Ends"][1])
                annotations["cDNA"]["Starts"] = [annotations["cDNA"]["Starts"][0]]
                annotations["cDNA"]["Ends"] = [annotations["cDNA"]["Ends"][0]]
            elif order == "-" and collapsed_array[-1] == "cDNA":
                annotations["random_s"]["Starts"].append(annotations["cDNA"]["Starts"][1])
                annotations["random_s"]["Ends"].append(annotations["cDNA"]["Ends"][1])
                annotations["cDNA"]["Starts"] = [annotations["cDNA"]["Starts"][0]]
                annotations["cDNA"]["Ends"] = [annotations["cDNA"]["Ends"][0]]
            elif order == "-" and collapsed_array[0] == "cDNA":
                annotations["random_e"]["Starts"].append(annotations["cDNA"]["Starts"][0])
                annotations["random_e"]["Ends"].append(annotations["cDNA"]["Ends"][0])
                annotations["cDNA"]["Starts"] = [annotations["cDNA"]["Starts"][1]]
                annotations["cDNA"]["Ends"] = [annotations["cDNA"]["Ends"][1]]

        if len(annotations["cDNA"]["Starts"]) == 3:
            if order == "+":
                annotations["random_s"]["Starts"].append(annotations["cDNA"]["Starts"][0])
                annotations["random_s"]["Ends"].append(annotations["cDNA"]["Ends"][0])
                annotations["random_e"]["Starts"].append(annotations["cDNA"]["Starts"][2])
                annotations["random_e"]["Ends"].append(annotations["cDNA"]["Ends"][2])
                annotations["cDNA"]["Starts"] = [annotations["cDNA"]["Starts"][1]]
                annotations["cDNA"]["Ends"] = [annotations["cDNA"]["Ends"][1]]
            else:
                annotations["random_e"]["Starts"].append(annotations["cDNA"]["Starts"][0])
                annotations["random_e"]["Ends"].append(annotations["cDNA"]["Ends"][0])
                annotations["random_s"]["Starts"].append(annotations["cDNA"]["Starts"][2])
                annotations["random_s"]["Ends"].append(annotations["cDNA"]["Ends"][2])
                annotations["cDNA"]["Starts"] = [annotations["cDNA"]["Starts"][1]]
                annotations["cDNA"]["Ends"] = [annotations["cDNA"]["Ends"][1]]

    annotations["architecture"] = "valid" if order_match else "invalid"
    annotations["read_length"] = str(read_length)
    annotations["orientation"] = order
    annotations["reason"] = reasons

    if annotations["architecture"] == "valid":
        for barcode in barcodes:
            annotations[barcode]["Sequences"] = [
                read[int(annotations[barcode]["Starts"][0]) : int(annotations[barcode]["Ends"][0])]
            ]

    # Compute edit distances for segments with known literal patterns (all reads, not just valid)
    if known_patterns:
        for seg, pattern in known_patterns.items():
            if seg in annotations and annotations[seg]["Starts"]:
                rc_pattern = reverse_complement(pattern)
                seqs = []
                dists = []
                for s, e in zip(annotations[seg]["Starts"], annotations[seg]["Ends"]):
                    seg_seq = read[int(s) : int(e)]
                    seqs.append(seg_seq)
                    dist_fwd = Levenshtein.distance(seg_seq, pattern)
                    dist_rc = Levenshtein.distance(seg_seq, rc_pattern)
                    dists.append(min(dist_fwd, dist_rc))
                annotations[seg]["Sequences"] = seqs
                annotations[seg]["EditDist"] = dists

    return [annotations]


def extract_annotated_full_length_seqs(
    new_data,
    predictions,
    model_path,
    read_lengths,
    label_binarizer,
    seq_order,
    barcodes,
    n_jobs,
    original_read_names=None,
    split_concatenated=False,
    valid_structures=None,
    known_patterns=None,
):
    """Extract annotated sequences from predictions using label binarizer and seq order."""
    if valid_structures is None:
        valid_structures = [seq_order]
    data = [(new_data[i], predictions[i], read_lengths[i], seq_order, valid_structures) for i in range(len(new_data))]

    raw_results = []

    if n_jobs == 1:
        for i in range(len(data)):
            raw_results.append(
                process_full_len_reads(
                    data[i], barcodes, label_binarizer, model_path, split_concatenated, known_patterns
                )
            )

    elif n_jobs > 1:
        with mp.Pool(processes=n_jobs) as pool:
            raw_results = pool.starmap(
                process_full_len_reads,
                [(d, barcodes, label_binarizer, model_path, split_concatenated, known_patterns) for d in data],
            )
            pool.close()

    # Flatten: each entry in raw_results is a list of annotation dicts
    # (1 for normal/invalid reads, N valid fragments + 1 invalid remainder for split reads)
    annotated_data = []
    expanded_read_names = []
    source_indices = []
    for i, result_list in enumerate(raw_results):
        base_name = original_read_names[i] if original_read_names is not None else f"read_{i}"
        if len(result_list) > 1:
            # Split concatenated read: valid fragments + invalid remainder (last entry)
            frag_counter = 1
            for ann in result_list:
                annotated_data.append(ann)
                if ann.get("architecture") == "invalid":
                    suffix = "__remainder"
                else:
                    suffix = f"__frag{frag_counter}"
                    frag_counter += 1
                expanded_read_names.append(f"{base_name}{suffix}")
                source_indices.append(i)
        else:
            annotated_data.append(result_list[0])
            expanded_read_names.append(base_name)
            source_indices.append(i)

    del data
    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()
    return annotated_data, expanded_read_names, source_indices
