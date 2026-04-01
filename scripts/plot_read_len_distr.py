import os
import logging
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_read_len_distr(parquet_dir, output_dir):
    """Generate and save a read-length distribution histogram."""
    os.makedirs(output_dir, exist_ok=True)
    parquet_files = [
        os.path.join(parquet_dir, f)
        for f in os.listdir(parquet_dir)
        if f.endswith(".parquet") and f != "read_index.parquet"
    ]

    if not parquet_files:
        logger.info("No Parquet files found in the directory.")
        return

    read_lengths = []
    for parquet_file in parquet_files:
        try:
            df = pl.read_parquet(parquet_file, columns=["read_length"])
            df = df.with_columns(pl.col("read_length").cast(pl.Int64, strict=False).alias("read_length")).filter(
                pl.col("read_length").is_not_null()
            )
            read_lengths.extend(df["read_length"].to_list())

        except Exception as e:
            logger.info(f"Error reading {parquet_file}: {e}")
            continue

    if not read_lengths:
        logger.info("No read lengths found.")
        return

    read_lengths = np.array(read_lengths, dtype=int)
    if len(read_lengths) > 0:
        logger.info(f"Minimum read length: {read_lengths.min()}, Maximum read length: {read_lengths.max()}")
    else:
        logger.info("No valid read lengths found after loading.")
        return

    # log_read_lengths = np.log10(read_lengths[read_lengths > 0])
    read_lengths = read_lengths[(read_lengths > 0) & (read_lengths < 6000)]

    # Plot the read length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(read_lengths, bins=100, color="skyblue", edgecolor="black", alpha=0.7)
    # plt.hist(log_read_lengths, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Read Length Distribution")
    plt.xlabel("Read Length")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Save the plot to the output directory
    plot_file = os.path.join(output_dir, "read_length_distribution.png")
    from utils import get_version
    plt.savefig(plot_file, metadata={"Software": f"tranquillyzer v{get_version()}"})
    plt.close()

    logger.info(f"Read length distribution plot saved to {plot_file}")
