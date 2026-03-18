import logging
import os

import yaml

logger = logging.getLogger(__name__)


def _load_yaml(file_path):
    """Load and return the contents of a YAML file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with open(file_path) as f:
        return yaml.safe_load(f)


def seq_orders(file_path, model):
    """Load sequence order config from YAML.

    Returns (seq_order, sequences, barcodes, umis, strand) — same 5-tuple as before.
    """
    config = _load_yaml(file_path)

    if model not in config:
        available = [k for k in config.keys()]
        raise ValueError(f"Model '{model}' not found in {file_path}. Available: {available}")

    entry = config[model]
    seq_order = [s["name"] for s in entry["segments"]]
    sequences = [s["pattern"] for s in entry["segments"]]
    barcodes = entry.get("barcodes", [])
    umis = entry.get("umis", [])
    strand = entry.get("strand", "")

    return seq_order, sequences, barcodes, umis, strand


def get_valid_structures(file_path, model):
    """Load valid structures for inference validation.

    Returns list of lists — each is an acceptable segment order.
    If not defined, defaults to the full segment order from 'segments'.
    """
    config = _load_yaml(file_path)

    if model not in config:
        raise ValueError(f"Model '{model}' not found in {file_path}")

    entry = config[model]
    structures = entry.get("valid_structures", None)
    if structures is None:
        structures = [[s["name"] for s in entry["segments"]]]
    return structures


def get_training_structures(file_path, model):
    """Load training structures with proportions and repeat counts.

    Returns list of dicts: [{order, patterns, repeat, proportion}, ...].
    Patterns are resolved from the segment vocabulary.
    If not defined, defaults to 100% of the full segment order.
    """
    config = _load_yaml(file_path)

    if model not in config:
        raise ValueError(f"Model '{model}' not found in {file_path}")

    entry = config[model]
    structs = entry.get("training_structures", None)

    if structs is None:
        return [{
            "order": [s["name"] for s in entry["segments"]],
            "patterns": [s["pattern"] for s in entry["segments"]],
            "repeat": 1,
            "proportion": 1.0,
        }]

    pattern_map = {s["name"]: s["pattern"] for s in entry["segments"]}
    result = []
    for s in structs:
        # Parse :rev/:fwd/:reverse suffixes from element names
        order_raw = s["order"]
        order = []
        rc_elements = {}
        for name in order_raw:
            if name.endswith(":reverse"):
                clean = name[:-8]
                order.append(clean)
                rc_elements[clean] = "reverse"
            elif name.endswith(":rev"):
                clean = name[:-4]
                order.append(clean)
                rc_elements[clean] = "rev"
            elif name.endswith(":fwd"):
                clean = name[:-4]
                order.append(clean)
                rc_elements[clean] = "fwd"
            else:
                order.append(name)
        patterns = [pattern_map[name] for name in order]
        repeat = s.get("repeat", 1)
        rc_pattern = s.get("rc_pattern", ["fwd"] * repeat)
        if len(rc_pattern) != repeat:
            raise ValueError(
                f"rc_pattern length ({len(rc_pattern)}) must match repeat ({repeat}) "
                f"in training structure: {s}"
            )
        for val in rc_pattern:
            if val not in ("fwd", "rev", "reverse"):
                raise ValueError(
                    f"rc_pattern values must be 'fwd', 'rev', or 'reverse', got '{val}' "
                    f"in training structure: {s}"
                )
        result.append({
            "order": order,
            "patterns": patterns,
            "repeat": repeat,
            "rc_elements": rc_elements,
            "rc_pattern": rc_pattern,
            "proportion": s["proportion"],
        })
    return result


def get_training_params(file_path, model):
    """Load training params from YAML.

    Returns dict of parameters. Values may be lists for grid search.
    """
    config = _load_yaml(file_path)

    if model not in config:
        raise ValueError(f"Model '{model}' not found in {file_path}")

    return config[model]


def trained_models():
    """Print a formatted table of available pre-trained models."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "..", "models")
    models_dir = os.path.abspath(models_dir)

    utils_dir = os.path.join(base_dir, "..", "utils")
    utils_dir = os.path.abspath(utils_dir)

    try:
        if not os.path.isdir(models_dir):
            print(f"The directory '{models_dir}' does not exist.")
            return

        print("\n~~~~~~~~~~~~~~~~ CURRENTLY AVAILABLE TRAINED MODELS ~~~~~~~~~~~~~~~~")
        print(
            "\n".join(
                [
                    "-- Sequence Key:",
                    "\tNX ==> unknown sequence of length X",
                    "\tNN ==> unknown sequence of unknown length",
                    "\tA  ==> sequence of A's of unknown length",
                    "\tT  ==> sequence of T's of unknown length",
                    "",
                ]
            )
        )

        seq_orders_file = os.path.join(utils_dir, "seq_orders.yaml")

        for file_name in os.listdir(models_dir):
            if file_name.endswith(".h5"):
                try:
                    seq_order, sequences, barcodes, UMIs, orientation = seq_orders(
                        seq_orders_file, file_name[:-3]
                    )

                    longest = max([len(x) for x in seq_order])

                    print_elements = [f"-- {file_name[:-3]}", "\tlayout (top to bottom) ==> sequence"]

                    for i in range(len(seq_order)):
                        print_elements.append(f"\t{seq_order[i]:<{longest}} ==> {sequences[i]}")

                    print_elements.append("")

                    print("\n".join(print_elements))
                except Exception:
                    print(
                        f"-- {file_name[:-3]}\n\t==> model exists in models/ directory but is undefined in utils/seq_orders.yaml\n"
                    )

    except Exception as e:
        print(f"An error occurred: {e}")
