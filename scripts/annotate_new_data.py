import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import gc
import time
import json
import yaml
import logging
import contextlib
import numpy as np
from typing import Optional
from numba import njit
import polars as pl
import tensorflow as tf

from scripts.train_new_model import ont_read_annotator
import scripts.available_gpus as available_gpus
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

# TODO: Will need to think about how to handle this when users are able to pick
#       which specific GPUs to use. This may need to be set elsewhere, or still
#       set here, but using a different function

from scripts.available_gpus import gpus_to_visible_devices_string

os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_visible_devices_string()

tf.config.experimental.enable_tensor_float_32_execution(False)
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

logger = logging.getLogger(__name__)

# Create a NumPy lookup array (256 elements to handle all ASCII characters)
NUCLEOTIDE_TO_ID = np.zeros(256, dtype=np.int32)
NUCLEOTIDE_TO_ID[ord("A")] = 1
NUCLEOTIDE_TO_ID[ord("C")] = 2
NUCLEOTIDE_TO_ID[ord("G")] = 3
NUCLEOTIDE_TO_ID[ord("T")] = 4
NUCLEOTIDE_TO_ID[ord("N")] = 5  # Default encoding for unknown nucleotides

# Enable memory growth to avoid pre-allocating all GPU memory
if available_gpus.n_gpus() > 0:
    for gpu in available_gpus.get_tensorflow_output():
        tf.config.experimental.set_memory_growth(gpu, True)

tf.config.optimizer.set_jit(True)


@njit
def encode_sequence_numba(read, encoded_seq):
    """nucleotide encoding using ASCII lookup."""
    for i in range(len(read)):
        encoded_seq[i] = NUCLEOTIDE_TO_ID[ord(read[i])]
    return encoded_seq


def preprocess_sequences(sequences, max_len):
    """Converts DNA sequences into NumPy integer arrays."""
    # max_len = max(len(seq) for seq in sequences)  # Get max sequence length
    encoded_array = np.zeros((len(sequences), max_len), dtype=np.int32)  # Pre-allocate array

    for i, seq in enumerate(sequences):
        encoded_array[i, : len(seq)] = encode_sequence_numba(seq, encoded_array[i, : len(seq)])

    return encoded_array


# Function to calculate the total number of rows in the Parquet file
def calculate_total_rows(parquet_file):
    """Return the total number of rows in a Parquet file."""
    df = pl.scan_parquet(parquet_file)
    total_rows = df.collect().shape[0]
    return total_rows


# Modified function to estimate the average read length from the bin name
def estimate_average_read_length_from_bin(bin_name):
    """Estimate average read length from a bin name like '100_500bp'."""
    bounds = bin_name.replace("bp", "").split("_")
    lower_bound = int(bounds[0])
    upper_bound = int(bounds[1])
    return (lower_bound + upper_bound) / 2


def num_replicas(strategy=None):
    """Return the number of replicas in a distribute strategy, or 1."""
    return getattr(strategy, "num_replicas_in_sync", 1) if strategy else 1


def bytes_from_gb(gb: float) -> int:
    """Convert gigabytes to bytes."""
    return int(float(gb) * (1024**3))


def parse_gpu_total_gb(user_total_gb: Optional[str], num_gpus: int) -> list[int]:
    """
    Accepts:
      - None -> default 12 GB per GPU
      - "48" -> broadcast 48 GB to all GPUs
      - "48,48,24" -> per-GPU totals (len must match or first is broadcast)
    Returns list of bytes per GPU.
    """
    default_gb = 12.0
    if not user_total_gb:
        return [bytes_from_gb(default_gb) for _ in range(num_gpus)]
    parts = [p.strip() for p in user_total_gb.split(",") if p.strip()]
    if len(parts) == 1:
        gb = float(parts[0])
        return [bytes_from_gb(gb) for _ in range(num_gpus)]
    vals = [float(p) for p in parts]
    if len(vals) != num_gpus:
        # Length mismatch: broadcast the first value
        return [bytes_from_gb(vals[0]) for _ in range(num_gpus)]
    return [bytes_from_gb(v) for v in vals]


def usable_bytes_per_gpu(total_bytes_per_gpu: list[int], safety_margin: float = 0.35) -> list[int]:
    """
    usable ≈ (total - TF_current) * (1 - safety_margin), clamped to >= 0
    """
    out = []
    handles = available_gpus.get_gpu_names_clean()
    for i, total in enumerate(total_bytes_per_gpu):
        try:
            info = tf.config.experimental.get_memory_info(handles[i])  # {'current','peak'}
            current = int(info.get("current", 0))
        except Exception:
            current = 0
        free_like = max(0, total - current)
        usable = int(max(0, free_like * (1.0 - safety_margin)))
        out.append(usable)
    return out


def pick_per_replica_batch_by_tokens(seq_len, target_tokens_per_replica=1_200_000, min_b=1, max_b=8192):
    """Compute per-replica batch size from a target token budget."""
    if seq_len <= 0:
        return min_b
    b = target_tokens_per_replica // int(seq_len)
    return int(max(min_b, min(max_b, b)))


def estimate_bytes_per_token(
    embedding_dim=128, conv_filters=128, conv_layers=3,
    lstm_units=96, bidirectional=True, num_labels=10,
    overhead_factor=1.12,
):
    """Estimate GPU memory per token from model architecture params.

    Accounts for embedding, conv activations, LSTM outputs + gate activations,
    and output/CRF layer.  *overhead_factor* covers TF bookkeeping (~12%).
    """
    bidir = 2 if bidirectional else 1
    cf = sum(conv_filters) if isinstance(conv_filters, list) else int(conv_filters) * int(conv_layers)
    lu = sum(lstm_units) if isinstance(lstm_units, list) else int(lstm_units)
    bpt = (
        4                       # input int32
        + embedding_dim * 4     # embedding
        + cf * 4                # conv layer outputs
        + lu * bidir * 4        # LSTM outputs
        + lu * 4 * bidir * 4    # LSTM gate activations
        + num_labels * 4        # CRF/output
    )
    return bpt * overhead_factor


def pick_per_replica_batch_by_model(usable_bytes_per_gpu, seq_len, bytes_per_token, min_b=1, max_b=8192):
    """For each GPU: B <= usable / (bytes_per_token * seq_len).

    Returns the minimum across GPUs (most constrained device).
    """
    caps = []
    denom = max(1, int(bytes_per_token * int(seq_len)))
    for usable in usable_bytes_per_gpu:
        if usable <= 0:
            caps.append(min_b)
        else:
            caps.append(int(max(min_b, min(max_b, usable // denom))))
    return min(caps) if caps else min_b


def choose_global_batch(
    L,
    bytes_per_token,
    strategy=None,
    target_tokens_per_replica=1_200_000,
    min_b=1,
    max_b=8192,
    user_total_gb: Optional[str] = None,
    safety_margin: float = 0.35,
    token_cap_above: int = 0,
):
    """
    Choose per-replica batch size, then scale by replicas.

    Two-tier logic controlled by *token_cap_above*:
      - Bins shorter than threshold: GPU capacity only (maximize throughput).
      - Bins at or above threshold:  min(GPU capacity, token budget) (conservative).
    """
    n_gpus = available_gpus.n_gpus()
    if n_gpus == 0:
        # CPU-only fallback: always use token-based (no VRAM estimate available)
        per_replica = pick_per_replica_batch_by_tokens(L, target_tokens_per_replica, min_b, max_b)
        return per_replica  # global==per_replica on CPU

    totals = parse_gpu_total_gb(user_total_gb, num_gpus=n_gpus)
    usable = usable_bytes_per_gpu(totals, safety_margin=safety_margin)
    limit_gpu = pick_per_replica_batch_by_model(usable, L, bytes_per_token, min_b, max_b)

    if token_cap_above > 0 and L < token_cap_above:
        per_replica = limit_gpu  # GPU capacity only
    else:
        limit_tok = pick_per_replica_batch_by_tokens(L, target_tokens_per_replica, min_b, max_b)
        per_replica = max(min_b, min(limit_gpu, limit_tok))

    replicas = num_replicas(strategy)
    return max(1, per_replica * replicas)


def predict_with_backoff(model, build_dataset_fn, start_batch: int, min_batch: int = 1, rebuild_model_fn=None):
    """Run model.predict with exponential batch-size backoff on OOM errors.

    When *rebuild_model_fn* is provided, the model is reconstructed after
    clearing the TF session so that MirroredStrategy state is restored.
    Returns ``(predictions, final_batch_size, model)`` — the model reference
    may differ from the input if a rebuild occurred.
    """
    bs = int(start_batch)
    last_err = None
    while bs >= int(min_batch):
        try:
            ds = build_dataset_fn(bs)
            result = model.predict(ds, verbose=0)
            return result, bs, model
        except (tf.errors.ResourceExhaustedError, tf.errors.CancelledError, tf.errors.InternalError,
                tf.errors.UnknownError) as e:
            last_err = e
            new_bs = max(int(min_batch), bs // 2)
            logger.warning(f"OOM at batch={bs}. Retrying with batch={new_bs}...")
            K.clear_session()
            gc.collect()
            for dev in available_gpus.get_gpu_names_raw():
                try:
                    tf.config.experimental.reset_memory_stats(dev)
                except Exception:
                    pass
            time.sleep(1.0)
            if rebuild_model_fn is not None:
                try:
                    model = rebuild_model_fn()
                    logger.info("Model rebuilt after OOM recovery")
                except Exception as rebuild_err:
                    logger.error(f"Failed to rebuild model after OOM: {rebuild_err}")
            bs = new_bs
    raise RuntimeError("Prediction OOM/cancelled even at batch=1") from last_err


_batch_log_once = set()


def _log_batch_once(bin_name: str, bs: int):
    """Log the batch size for a bin name only on first call."""
    if bin_name and bin_name not in _batch_log_once:
        logger.info(f"Using batch_size={bs} for bin {bin_name}")
        _batch_log_once.add(bin_name)


def annotate_new_data_parallel(new_encoded_data, model, global_bs, min_batch=1, rebuild_model_fn=None):
    """Run model inference on encoded data using GPU or CPU fallback.

    Returns ``(predictions, model)`` — the model reference may change if an
    OOM triggered a rebuild via *rebuild_model_fn*.
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if available_gpus.n_gpus() == 0:
        cpu_bs = min(32, max(1, len(new_encoded_data)))  # small & predictable for tests/CI
        return model.predict(new_encoded_data, batch_size=cpu_bs, verbose=0), model

    def build_ds(bs: int):
        return (
            tf.data.Dataset.from_tensor_slices(new_encoded_data)
            .batch(int(bs), drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

    preds, _, model = predict_with_backoff(model, build_ds, global_bs, min_batch, rebuild_model_fn)
    return preds, model


def load_model_params(model_path):
    """Load model hyperparameters from the YAML or JSON params file alongside a model."""
    if not model_path:
        return None
    yaml_path = model_path.replace(".h5", "_params.yaml")
    json_path = model_path.replace(".h5", "_params.json")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    elif os.path.exists(json_path):
        with open(json_path) as f:
            raw_params = json.load(f)
        return {
            k: (
                v.lower() == "true"
                if isinstance(v, str) and v.lower() in ["true", "false"]
                else int(v)
                if isinstance(v, str) and v.isdigit()
                else float(v)
                if isinstance(v, str) and v.replace(".", "", 1).isdigit()
                else v
            )
            for k, v in raw_params.items()
        }
    else:
        raise FileNotFoundError(f"No params file found at {yaml_path} or {json_path}")


def build_model(model_path, conv_filters, num_labels, strategy=None):
    """Build a CRF Keras model from params and load weights, with optional distribution strategy."""
    params = load_model_params(model_path)

    # Normalize scalar-or-list params for backward compat with old params files
    p_conv_filters = params.get("conv_filters", conv_filters)
    if isinstance(p_conv_filters, (int, float)):
        p_conv_filters = [int(p_conv_filters)] * int(params["conv_layers"])
    conv_filters = max(p_conv_filters)  # scalar for batch-sizing callers

    conv_kernel_sizes = params.get("conv_kernel_sizes", params.get("conv_kernel_size", [25]))
    if isinstance(conv_kernel_sizes, (int, float)):
        conv_kernel_sizes = [int(conv_kernel_sizes)] * int(params["conv_layers"])

    p_lstm_units = params.get("lstm_units")
    if isinstance(p_lstm_units, (int, float)):
        p_lstm_units = [int(p_lstm_units) // (2**i) for i in range(int(params["lstm_layers"]))]

    scope = strategy.scope() if strategy else contextlib.nullcontext()
    with scope:
        model = ont_read_annotator(
            vocab_size=params["vocab_size"],
            embedding_dim=params["embedding_dim"],
            num_labels=num_labels,
            conv_layers=params["conv_layers"],
            conv_filters=p_conv_filters,
            conv_kernel_sizes=conv_kernel_sizes,
            dilation_rates=params.get("dilation_rates"),
            lstm_layers=params["lstm_layers"],
            lstm_units=p_lstm_units,
            bidirectional=params["bidirectional"],
            attention_heads=params["attention_heads"],
            dropout_rate=params["dropout_rate"],
            regularization=params["regularization"],
            learning_rate=params["learning_rate"],
            crf_layer=True,
        )
    _ = model(tf.zeros((1, 512), dtype=tf.int32), training=False)
    model.load_weights(model_path)
    return model


def load_model_for_inference(model_path, num_labels):
    """Build the CRF model once for reuse across all bins.

    Returns ``(model, strategy, params, min_batch)`` so the caller can
    pass them into :func:`model_predictions` for each bin.
    """
    n_gpus = available_gpus.n_gpus()
    min_batch = int(n_gpus) if n_gpus > 0 else 1
    strategy = tf.distribute.MirroredStrategy() if n_gpus > 1 else None

    conv_filters = 256
    params = load_model_params(model_path)
    if params:
        cf = params.get("conv_filters", conv_filters)
        conv_filters = max(cf) if isinstance(cf, list) else int(cf)

    model = build_model(model_path, conv_filters, num_labels, strategy=strategy)
    return model, strategy, params, min_batch


def model_predictions(
    parquet_file,
    chunk_start,
    chunk_size,
    model,
    model_path,
    strategy,
    params,
    num_labels,
    user_total_gb: Optional[str] = None,  # e.g. "48" or "48,48,24"; None → 12 GB/GPU
    target_tokens_per_replica: int = 1_200_000,  # tune 0.8–1.5M per your GPUs
    safety_margin: float = 0.35,  # keep ~35% VRAM headroom
    min_batch: int = 1,
    max_batch: int = 8192,
    token_cap_above: int = 0,
    should_process_chunk=None,
):
    """Yield per-chunk (predictions, read_names, reads, lengths, qualities) from a Parquet file.

    The *model* is pre-built by the caller via :func:`load_model_for_inference`
    and reused across bins.
    """
    total_rows = calculate_total_rows(parquet_file)
    bin_name = os.path.basename(parquet_file).replace(".parquet", "")

    # Estimate the average read length from the bin name and adjust chunk size
    estimated_avg_length = estimate_average_read_length_from_bin(bin_name)
    dynamic_chunk_size = int(chunk_size * (500 / estimated_avg_length))
    dynamic_chunk_size = min(dynamic_chunk_size, 500000)

    scan_df = pl.scan_parquet(parquet_file)
    num_chunks = (total_rows // dynamic_chunk_size) + (1 if total_rows % dynamic_chunk_size > 0 else 0)

    conv_filters = 256
    if params:
        cf = params.get("conv_filters", conv_filters)
        conv_filters = max(cf) if isinstance(cf, list) else int(cf)
        bpt = estimate_bytes_per_token(
            embedding_dim=int(params.get("embedding_dim", 128)),
            conv_filters=params.get("conv_filters", 256),
            conv_layers=int(params.get("conv_layers", 3)),
            lstm_units=params.get("lstm_units", 96),
            bidirectional=bool(params.get("bidirectional", True)),
            num_labels=num_labels,
        )
    else:
        bpt = estimate_bytes_per_token(num_labels=num_labels)

    # Add one-sided receptive field of the largest dilated conv layer as padding
    # so the model can properly predict end-of-sequence transitions for all reads in the bin.
    if params:
        kernel_sizes = params.get("conv_kernel_sizes", [25, 25, 25])
        dilation_rates = params.get("dilation_rates", [1, 1, 1])
        conv_padding = max((k - 1) * d for k, d in zip(kernel_sizes, dilation_rates)) // 2
    else:
        conv_padding = 12  # default: half of kernel_size=25
    max_len = int(bin_name.replace("bp", "").split("_")[1]) + 1 + conv_padding
    global_bs = choose_global_batch(
        max_len,
        bytes_per_token=bpt,
        strategy=strategy,
        target_tokens_per_replica=target_tokens_per_replica,
        min_b=min_batch,
        max_b=max_batch,
        user_total_gb=user_total_gb,
        safety_margin=safety_margin,
        token_cap_above=token_cap_above,
    )

    def _rebuild_model():
        return build_model(model_path, conv_filters, num_labels, strategy=strategy)

    using_strategy = strategy is not None

    for chunk_idx in range(chunk_start, num_chunks + 1):
        if callable(should_process_chunk) and not should_process_chunk(bin_name, chunk_idx):
            logger.info(f"Skipping {bin_name}: chunk {chunk_idx} (already completed)")
            continue

        logger.info(f"Inferring labels for {bin_name}: chunk {chunk_idx}")

        df_chunk = scan_df.slice((chunk_idx - 1) * dynamic_chunk_size, dynamic_chunk_size).collect()
        read_names = df_chunk["ReadName"].to_list()
        reads = df_chunk["read"].to_list()
        read_lengths = df_chunk["read_length"].to_list()
        base_qualities = df_chunk["base_qualities"].to_list() if "base_qualities" in df_chunk.columns else None

        encoded_data = preprocess_sequences(reads, max_len)
        X_new_padded = pad_sequences(encoded_data, padding="post", dtype="int32", maxlen=max_len)

        _log_batch_once(bin_name or "", int(global_bs))

        use_strategy = global_bs >= 100 and len(reads) >= 100
        if use_strategy:
            if not using_strategy and strategy is not None:
                model = build_model(model_path, conv_filters, num_labels, strategy=strategy)
                using_strategy = True
            chunk_predictions, model = annotate_new_data_parallel(
                X_new_padded, model, global_bs, min_batch=min_batch,
                rebuild_model_fn=_rebuild_model,
            )
        else:
            if using_strategy:
                model = build_model(model_path, conv_filters, num_labels, strategy=None)
                using_strategy = False
            small_bs = max(1, min(global_bs, len(reads)))
            chunk_predictions = model.predict(X_new_padded, batch_size=small_bs, verbose=0)

        del df_chunk, X_new_padded
        gc.collect()
        logger.info(f"Inferred labels for {bin_name}: chunk {chunk_idx}")

        yield (
            parquet_file,
            bin_name,
            chunk_idx,
            chunk_predictions,
            read_names,
            reads,
            read_lengths,
            base_qualities,
        )
