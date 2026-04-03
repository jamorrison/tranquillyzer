import re
import random
import logging
import numpy as np
from multiprocessing import Pool

logger = logging.getLogger(__name__)

############## introduce errors ##############


def introduce_errors_with_labels_context(
    sequence, label, mismatch_rate, insertion_rate, deletion_rate, polyT_error_rate, max_insertions
):
    """Introduce substitution/insertion/deletion errors into a sequence while tracking label offsets."""
    error_sequence, error_labels = [], []
    for base, lbl in zip(sequence, label):
        insertion_count = 0
        r = np.random.random()

        if lbl in ("polyT", "polyA"):
            local_mismatch_rate = polyT_error_rate
            local_insertion_rate = polyT_error_rate
            local_deletion_rate = polyT_error_rate
        elif lbl == "ACC" or lbl == "cDNA":
            local_mismatch_rate = 0
            local_insertion_rate = 0
            local_deletion_rate = 0
        else:
            local_mismatch_rate = mismatch_rate
            local_insertion_rate = insertion_rate
            local_deletion_rate = deletion_rate

        if r < local_mismatch_rate:
            error_sequence.append(np.random.choice([b for b in "ATCG" if b != base]))
            error_labels.append(lbl)
        elif r < local_mismatch_rate + local_insertion_rate:
            error_sequence.append(base)
            error_labels.append(lbl)
            while insertion_count < max_insertions:
                error_sequence.append(np.random.choice(list("ATCG")))
                error_labels.append(lbl)
                insertion_count += 1
                if np.random.random() >= local_insertion_rate:
                    break
        elif r < local_mismatch_rate + local_insertion_rate + local_deletion_rate:
            continue
        else:
            error_sequence.append(base)
            error_labels.append(lbl)

    return "".join(error_sequence), error_labels


############## generate segments ##############


def generate_segment(segment_type, segment_pattern, length_range, transcriptome_records, spacer_range=(0, 50)):
    """Generate a random DNA segment matching a pattern specification."""
    if re.match(r"N\d+", segment_pattern):
        length = int(segment_pattern[1:])
        sequence = "".join(np.random.choice(list("ATCG")) for _ in range(length))
        label = [segment_type] * length
    elif segment_pattern == "NN" and segment_type == "cDNA":
        length = np.random.randint(length_range[0], length_range[1])
        if transcriptome_records:
            transcript = random.choice(transcriptome_records)
            transcript_seq = str(transcript.seq) if hasattr(transcript, "seq") else str(transcript)
        else:
            transcript_seq = "".join(np.random.choice(list("ATCG")) for _ in range(length))
        fragment = (
            transcript_seq[:length]
            if len(transcript_seq) > length and random.random() < 0.5
            else transcript_seq[-length:]
        )
        sequence = fragment
        label = ["cDNA"] * len(sequence)
    elif segment_pattern == "RN" and segment_type == "cDNA":
        length = np.random.randint(spacer_range[0], spacer_range[1] + 1)
        length = min(length, 50)
        if length == 0:
            return "", []
        if transcriptome_records:
            transcript = random.choice(transcriptome_records)
            transcript_seq = str(transcript.seq) if hasattr(transcript, "seq") else str(transcript)
        else:
            transcript_seq = "".join(np.random.choice(list("ATCG")) for _ in range(length))
        sequence = transcript_seq[:length]
        label = ["cDNA"] * len(sequence)
    elif segment_pattern in ["A", "T"]:
        length = np.random.randint(0, 50)
        sequence = segment_pattern * length
        label = [segment_type] * length
    else:
        sequence = segment_pattern
        label = [segment_type] * len(sequence)
    return sequence, label


############## generate valid read ##############


def generate_valid_read(segments_order, segments_patterns, length_range, transcriptome_records, spacer_range=(0, 50)):
    """Assemble a full synthetic read from segment patterns and structure order."""
    read_segments, label_segments = [], []
    for seg_type, seg_pattern in zip(segments_order, segments_patterns):
        s, labs = generate_segment(seg_type, seg_pattern, length_range, transcriptome_records, spacer_range)
        read_segments.append(s)
        label_segments.append(labs)
    return "".join(read_segments), [lbl for seg_lbls in label_segments for lbl in seg_lbls]


############## reverse complement utilities ##############


def reverse_complement(sequence):
    """Return the reverse complement of a DNA sequence."""
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(complement[base] for base in reversed(sequence))


def reverse_labels(labels):
    """Reverse a list of per-position labels."""
    return labels[::-1]


############## config-driven training data generation ##############


_RC_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def _rc_pattern_str(pattern):
    """Reverse-complement a literal adapter pattern string."""
    return "".join(_RC_COMPLEMENT[b] for b in reversed(pattern))


def _rc_single_pattern(pattern):
    """Reverse-complement a single element's pattern based on its type."""
    if pattern in ("A", "T"):
        return "T" if pattern == "A" else "A"
    elif re.match(r"N\d+", pattern) or pattern in ("NN", "RN"):
        return pattern
    else:
        return _rc_pattern_str(pattern)


def _reverse_single_pattern(pattern):
    """Reverse (without complement) a single element's pattern based on its type."""
    if pattern in ("A", "T"):
        # Poly patterns unchanged when just reversed
        return pattern
    elif re.match(r"N\d+", pattern) or pattern in ("NN", "RN"):
        return pattern
    else:
        # Literal adapter — reverse the string only (no complement)
        return pattern[::-1]


# Maps (current_state, desired_state) to the transform needed.
# States: "fwd", "rev" (reverse-complement), "reverse" (reverse-only)
def _transform_pattern(pattern, from_state, to_state):
    """Transform a pattern between orientation states."""
    if from_state == to_state:
        return pattern
    # Build lookup of transforms
    if from_state == "fwd" and to_state == "rev":
        return _rc_single_pattern(pattern)
    elif from_state == "fwd" and to_state == "reverse":
        return _reverse_single_pattern(pattern)
    elif from_state == "rev" and to_state == "fwd":
        return _rc_single_pattern(pattern)  # RC is self-inverse
    elif from_state == "rev" and to_state == "reverse":
        # RC'd → fwd → reverse: complement only (undo reverse, keep complement undone... )
        # Actually: RC = reverse + complement. To go from RC to reverse-only,
        # we need to undo the complement: apply complement without reversing.
        return "".join(_RC_COMPLEMENT.get(b, b) for b in pattern) if not (
            pattern in ("A", "T") or re.match(r"N\d+", pattern) or pattern in ("NN", "RN")
        ) else pattern
    elif from_state == "reverse" and to_state == "fwd":
        return _reverse_single_pattern(pattern)  # reverse is self-inverse
    elif from_state == "reverse" and to_state == "rev":
        # reverse → fwd → RC
        fwd = _reverse_single_pattern(pattern)
        return _rc_single_pattern(fwd)
    return pattern


def _build_fragment(order, patterns, fragment_orientation, rc_elements):
    """Build a fragment's order and patterns, applying fragment-level and element-level orientation.

    Args:
        order: list of element names
        patterns: list of patterns corresponding to order
        fragment_orientation: "fwd", "rev" (reverse-complement), or "reverse" (reverse-only)
        rc_elements: dict of element-level overrides, e.g. {"3p": "rev", "5p": "fwd", "UMI": "reverse"}
    """
    if fragment_orientation in ("rev", "reverse"):
        # Reverse the order
        frag_order = order[::-1]
        if fragment_orientation == "rev":
            frag_patterns = [_rc_single_pattern(p) for p in reversed(patterns)]
        else:
            frag_patterns = [_reverse_single_pattern(p) for p in reversed(patterns)]
    else:
        frag_order = list(order)
        frag_patterns = list(patterns)

    # Apply per-element overrides (absolute: override whatever fragment-level did)
    for i, name in enumerate(frag_order):
        override = rc_elements.get(name)
        if override is None:
            continue
        frag_patterns[i] = _transform_pattern(
            frag_patterns[i], fragment_orientation, override
        )

    return frag_order, frag_patterns


def _build_structure_order_and_patterns(struct):
    """Build the full segment order and patterns for a training structure,
    including cDNA flanking and repeat handling."""
    order = struct["order"]
    patterns = struct["patterns"]
    repeat = struct.get("repeat", 1)
    rc_pattern = struct.get("rc_pattern", ["fwd"] * repeat)
    rc_elements = struct.get("rc_elements", {})

    if repeat > 1:
        # Concatenate: repeat the core with cDNA flanks between copies
        full_order = ["cDNA"]
        full_patterns = ["RN"]
        for i in range(repeat):
            frag_order, frag_patterns = _build_fragment(
                order, patterns, rc_pattern[i], rc_elements
            )
            full_order.extend(frag_order + ["cDNA"])
            full_patterns.extend(frag_patterns + ["RN"])
    else:
        # Single structure flanked with random cDNA
        frag_order, frag_patterns = _build_fragment(
            order, patterns, rc_pattern[0], rc_elements
        )
        full_order = ["cDNA"] + frag_order + ["cDNA"]
        full_patterns = ["RN"] + frag_patterns + ["RN"]

    return full_order, full_patterns


def _maybe_truncate(sequence, label, max_trunc_5p, max_trunc_3p):
    """Truncate ends that lack random flanking cDNA."""
    if not sequence:
        return sequence, label
    if max_trunc_5p > 0 and label[0] != "cDNA":
        t = random.randint(0, max_trunc_5p)
        sequence, label = sequence[t:], label[t:]
    if max_trunc_3p > 0 and sequence and label[-1] != "cDNA":
        t = random.randint(0, max_trunc_3p)
        if 0 < t < len(sequence):
            sequence, label = sequence[:-t], label[:-t]
    return sequence, label


def simulate_dynamic_batch_complete(
    num_reads,
    length_range,
    mismatch_rate,
    insertion_rate,
    deletion_rate,
    polyT_error_rate,
    max_insertions,
    training_structures,
    transcriptome_records,
    rc,
    max_trunc_5p=0,
    max_trunc_3p=0,
    min_spacer=0,
    max_spacer=50,
):
    """Simulate a batch of synthetic reads with dynamic length and error profiles."""
    reads, labels, expected_fragments = [], [], []
    weights = [s["proportion"] for s in training_structures]

    for _ in range(num_reads):
        struct = random.choices(training_structures, weights=weights, k=1)[0]
        n_fragments = struct.get("repeat", 1)
        full_order, full_patterns = _build_structure_order_and_patterns(struct)

        struct_length_range = struct.get("length_range", length_range)
        sequence, label = generate_valid_read(full_order, full_patterns, struct_length_range, transcriptome_records, spacer_range=(min_spacer, max_spacer))

        sequence, label = _maybe_truncate(sequence, label, max_trunc_5p, max_trunc_3p)

        read_pairs = [(sequence, label)]
        if rc:
            rc_seq = reverse_complement(sequence)
            rc_lbl = reverse_labels(label)
            read_pairs.append((rc_seq, rc_lbl))

        for seq, lbl in read_pairs:
            seq_err, lbl_err = introduce_errors_with_labels_context(
                seq, lbl, mismatch_rate, insertion_rate, deletion_rate, polyT_error_rate, max_insertions
            )
            reads.append(seq_err)
            labels.append(lbl_err)
            expected_fragments.append(n_fragments)

    return reads, labels, expected_fragments


# ############## multiprocessing ##############


def simulate_dynamic_batch_complete_wrapper(args):
    """Wrapper for simulate_dynamic_batch_complete for use with multiprocessing."""
    return simulate_dynamic_batch_complete(*args)


def simulate_and_write_fasta(args):
    """Generate reads and write directly to a FASTA file. Returns (labels, expected_fragments) only.

    Args tuple: (num_reads, length_range, mismatch_rate, insertion_rate, deletion_rate,
                 polyT_error_rate, max_insertions, training_structures, transcriptome_records,
                 rc, max_trunc_5p, max_trunc_3p, min_spacer, max_spacer,
                 fasta_path, start_idx)
    """
    *sim_args, fasta_path, start_idx = args

    (num_reads, length_range, mismatch_rate, insertion_rate, deletion_rate,
     polyT_error_rate, max_insertions, training_structures, transcriptome_records,
     rc, max_trunc_5p, max_trunc_3p, min_spacer, max_spacer) = sim_args

    labels, expected_fragments, structure_names = [], [], []
    weights = [s["proportion"] for s in training_structures]
    read_idx = start_idx

    with open(fasta_path, "w") as fh:
        for _ in range(num_reads):
            struct = random.choices(training_structures, weights=weights, k=1)[0]
            n_fragments = struct.get("repeat", 1)
            struct_name = struct.get("name", "unknown")
            full_order, full_patterns = _build_structure_order_and_patterns(struct)

            struct_length_range = struct.get("length_range", length_range)
            sequence, label = generate_valid_read(
                full_order, full_patterns, struct_length_range, transcriptome_records,
                spacer_range=(min_spacer, max_spacer),
            )
            sequence, label = _maybe_truncate(sequence, label, max_trunc_5p, max_trunc_3p)

            read_pairs = [(sequence, label)]
            if rc:
                rc_seq = reverse_complement(sequence)
                rc_lbl = reverse_labels(label)
                read_pairs.append((rc_seq, rc_lbl))

            for seq, lbl in read_pairs:
                seq_err, lbl_err = introduce_errors_with_labels_context(
                    seq, lbl, mismatch_rate, insertion_rate, deletion_rate,
                    polyT_error_rate, max_insertions,
                )
                fh.write(f">assess_{read_idx}\n{seq_err}\n")
                labels.append(lbl_err)
                expected_fragments.append(n_fragments)
                structure_names.append(struct_name)
                read_idx += 1

    return labels, expected_fragments, structure_names


# ############## main generator ##############


def generate_training_reads(
    num_reads,
    mismatch_rate,
    insertion_rate,
    deletion_rate,
    polyT_error_rate,
    max_insertions,
    training_structures,
    length_range,
    num_processes,
    rc,
    transcriptome_records,
    max_trunc_5p=0,
    max_trunc_3p=0,
    min_spacer=0,
    max_spacer=50,
):
    """Generate a full set of synthetic training reads and labels."""
    # Convert BioPython SeqRecords to plain strings for fast pickling across workers
    transcriptome_seqs = [str(rec.seq) if hasattr(rec, "seq") else str(rec) for rec in transcriptome_records]

    effective_workers = max(1, min(num_processes, num_reads))

    base_chunk = num_reads // effective_workers
    remainder = num_reads % effective_workers
    chunks = [base_chunk + (1 if i < remainder else 0) for i in range(effective_workers)]

    worker_args = [
        (
            chunk_size,
            length_range,
            mismatch_rate,
            insertion_rate,
            deletion_rate,
            polyT_error_rate,
            max_insertions,
            training_structures,
            transcriptome_seqs,
            rc,
            max_trunc_5p,
            max_trunc_3p,
            min_spacer,
            max_spacer,
        )
        for chunk_size in chunks
    ]

    if effective_workers > 1:
        with Pool(processes=effective_workers) as pool:
            complete_results = pool.map(simulate_dynamic_batch_complete_wrapper, worker_args)
    else:
        complete_results = [simulate_dynamic_batch_complete_wrapper(worker_args[0])]

    reads, labels, expected_fragments = [], [], []
    for local_reads, local_labels, local_frags in complete_results:
        reads.extend(local_reads)
        labels.extend(local_labels)
        expected_fragments.extend(local_frags)

    return reads, labels, expected_fragments
