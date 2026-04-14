import copy
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


def _resolve_model(config, model):
    """Resolve a model entry by merging library defaults with model overrides.

    Supports both legacy flat format (no 'libraries' key) and the new
    two-level libraries/models format.

    Returns a flat dict with keys matching the legacy per-model structure:
    strand, barcodes, umis, segments, valid_structures, training_structures.
    """
    # Legacy format: no 'libraries' key means old flat dict keyed by model name
    if "libraries" not in config:
        if model not in config:
            available = list(config.keys())
            raise ValueError(f"Model '{model}' not found. Available: {available}")
        return config[model]

    # New two-level format
    models = config.get("models", {})
    libraries = config.get("libraries", {})

    if model not in models:
        available = list(models.keys())
        raise ValueError(f"Model '{model}' not found. Available models: {available}")

    model_entry = models[model]

    if "library" not in model_entry:
        raise ValueError(
            f"Model '{model}' must have a 'library' field referencing a library in the 'libraries' section."
        )

    lib_name = model_entry["library"]
    if lib_name not in libraries:
        available_libs = list(libraries.keys())
        raise ValueError(
            f"Library '{lib_name}' referenced by model '{model}' not found. Available libraries: {available_libs}"
        )

    resolved = copy.deepcopy(libraries[lib_name])

    # Simple fields: model value replaces library value entirely
    for field in ("strand", "barcodes", "umis", "segments", "valid_structures"):
        if field in model_entry:
            resolved[field] = model_entry[field]

    # Dict-level merge for training_structures and assessment_structures
    for struct_key in ("training_structures", "assessment_structures"):
        if struct_key in model_entry:
            model_ts = model_entry[struct_key]
            if model_ts is None:
                resolved[struct_key] = None
            else:
                lib_ts = resolved.get(struct_key)
                if lib_ts is None:
                    resolved[struct_key] = model_ts
                else:
                    merged = copy.deepcopy(lib_ts)
                    for name, value in model_ts.items():
                        if value is None:
                            merged.pop(name, None)
                        elif name in merged:
                            merged[name].update(value)
                        else:
                            merged[name] = value
                    resolved[struct_key] = merged

    return resolved


def seq_orders(file_path, model):
    """Load sequence order config from YAML.

    Returns (seq_order, sequences, barcodes, umis, strand) — same 5-tuple as before.
    """
    config = _load_yaml(file_path)
    entry = _resolve_model(config, model)

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
    entry = _resolve_model(config, model)

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
    entry = _resolve_model(config, model)
    ts = entry.get("training_structures", None)

    if ts is None:
        return [
            {
                "order": [s["name"] for s in entry["segments"]],
                "patterns": [s["pattern"] for s in entry["segments"]],
                "repeat": 1,
                "proportion": 1.0,
            }
        ]

    # Convert from named dict (new format) or list (legacy format)
    if isinstance(ts, dict):
        structs = list(ts.values())
    else:
        structs = ts

    # Validate proportions sum
    total = sum(s.get("proportion", 0) for s in structs)
    if abs(total - 1.0) > 0.01:
        logger.warning(f"Training structure proportions for model '{model}' sum to {total:.4f}, expected ~1.0")

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
                f"rc_pattern length ({len(rc_pattern)}) must match repeat ({repeat}) in training structure: {s}"
            )
        for val in rc_pattern:
            if val not in ("fwd", "rev", "reverse"):
                raise ValueError(
                    f"rc_pattern values must be 'fwd', 'rev', or 'reverse', got '{val}' in training structure: {s}"
                )
        result.append(
            {
                "order": order,
                "patterns": patterns,
                "repeat": repeat,
                "rc_elements": rc_elements,
                "rc_pattern": rc_pattern,
                "proportion": s["proportion"],
            }
        )
    return result


def get_assessment_structures(file_path, model):
    """Load assessment structures for model evaluation.

    Returns list of dicts: [{order, patterns, repeat, proportion, rc_pattern, rc_elements}, ...].
    Same format as get_training_structures but reads from 'assessment_structures'.
    Falls back to a single full-order structure if not defined.
    """
    config = _load_yaml(file_path)
    entry = _resolve_model(config, model)
    ts = entry.get("assessment_structures", None)

    if ts is None:
        return [
            {
                "name": "default",
                "order": [s["name"] for s in entry["segments"]],
                "patterns": [s["pattern"] for s in entry["segments"]],
                "repeat": 1,
                "proportion": 1.0,
            }
        ]

    if isinstance(ts, dict):
        struct_names = list(ts.keys())
        structs = list(ts.values())
    else:
        struct_names = [f"struct_{i}" for i in range(len(ts))]
        structs = ts

    total = sum(s.get("proportion", 0) for s in structs)
    if abs(total - 1.0) > 0.01:
        logger.warning(f"Assessment structure proportions for model '{model}' sum to {total:.4f}, expected ~1.0")

    pattern_map = {s["name"]: s["pattern"] for s in entry["segments"]}
    result = []
    for sname, s in zip(struct_names, structs):
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
                f"rc_pattern length ({len(rc_pattern)}) must match repeat ({repeat}) in assessment structure: {s}"
            )
        for val in rc_pattern:
            if val not in ("fwd", "rev", "reverse"):
                raise ValueError(
                    f"rc_pattern values must be 'fwd', 'rev', or 'reverse', got '{val}' in assessment structure: {s}"
                )
        result.append(
            {
                "name": sname,
                "order": order,
                "patterns": patterns,
                "repeat": repeat,
                "rc_elements": rc_elements,
                "rc_pattern": rc_pattern,
                "proportion": s["proportion"],
            }
        )
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
                    seq_order, sequences, barcodes, UMIs, orientation = seq_orders(seq_orders_file, file_name[:-3])

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
