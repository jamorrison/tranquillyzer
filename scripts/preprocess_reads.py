import os
import gc
import gzip
import logging
import time
import glob
import polars as pl
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor

from filelock import FileLock  # Import the FileLock library

logger = logging.getLogger(__name__)


def determine_bin(length, bin_size=500, adaptive_bin_threshold=10000):
    """Assign a read to a length bin based on configurable bin width.

    Below *adaptive_bin_threshold*, the user-supplied *bin_size* is used.
    Above it, fixed coarse tiers apply (5 000 / 10 000 / 25 000 bp).
    """
    if length < adaptive_bin_threshold:
        effective = bin_size
    elif length < 50000:
        effective = 5000
    elif length < 100000:
        effective = 10000
    else:
        effective = 25000
    bin_start = (length // effective) * effective
    bin_end = bin_start + effective - 1
    return f"{bin_start}_{bin_end}bp"


def extract_and_bin_reads(
    file_path, batch_size, output_dir, output_base_qual, bin_size=500, adaptive_bin_threshold=10000
):
    """Read sequences from a FASTA/FASTQ file and bin them by length."""
    reads_by_bin = {}
    file_format = "fasta" if file_path.endswith((".fa", ".fasta", ".fa.gz", ".fasta.gz")) else "fastq"

    with gzip.open(file_path, "rt") if file_path.endswith(".gz") else open(file_path, "r") as handle:
        for record in SeqIO.parse(handle, file_format):
            read_length = len(record.seq)
            bin_name = determine_bin(read_length, bin_size, adaptive_bin_threshold)

            # Ensure that the bin is initialized with all required keys
            if output_base_qual:
                if bin_name not in reads_by_bin:
                    reads_by_bin[bin_name] = {"read_names": [], "reads": [], "read_lengths": [], "base_quals": []}
                reads_by_bin[bin_name]["base_quals"].append(record.format("fastq").splitlines()[3])

            else:
                if bin_name not in reads_by_bin:
                    reads_by_bin[bin_name] = {"read_names": [], "reads": [], "read_lengths": []}

            reads_by_bin[bin_name]["read_names"].append(record.id)
            reads_by_bin[bin_name]["reads"].append(str(record.seq))
            reads_by_bin[bin_name]["read_lengths"].append(read_length)

            # Once a bin reaches the batch size, save it and reset
            if len(reads_by_bin[bin_name]["reads"]) >= batch_size:
                dump_bin_data(output_dir, output_base_qual, bin_name, reads_by_bin[bin_name])
                if output_base_qual:
                    reads_by_bin[bin_name] = {"read_names": [], "reads": [], "read_lengths": [], "base_quals": []}
                else:
                    reads_by_bin[bin_name] = {"read_names": [], "reads": [], "read_lengths": []}

        # Dump any remaining data after file is fully read
        for bin_name, data in reads_by_bin.items():
            if data["reads"]:
                dump_bin_data(output_dir, output_base_qual, bin_name, data)


def dump_bin_data(output_dir, output_base_qual, bin_name, data):
    """Write binned read data to TSV files in the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    tsv_filename = os.path.join(output_dir, f"{bin_name}.tsv")
    lock_filename = tsv_filename + ".lock"  # Create a lock file for the TSV

    if len(data["reads"]) == 0:
        return

    if output_base_qual:
        df = pl.DataFrame(
            {
                "ReadName": data["read_names"],
                "read": data["reads"],
                "read_length": data["read_lengths"],
                "base_qualities": data["base_quals"],
            }
        )
    else:
        df = pl.DataFrame({"ReadName": data["read_names"], "read": data["reads"], "read_length": data["read_lengths"]})

    try:
        # Use a file lock to prevent concurrent writes
        # Ensure only one process writes to this file at a time
        with FileLock(lock_filename):
            write_header = not os.path.exists(tsv_filename) or os.path.getsize(tsv_filename) == 0

            with open(tsv_filename, "a") as f:
                if write_header:
                    f.write("\t".join(df.columns) + "\n")
                for row in df.to_numpy():
                    f.write("\t".join(map(str, row)) + "\n")

    except Exception as e:
        logger.error(f"Error writing {tsv_filename}: {e}")


def parallel_preprocess_data(
    file_list,
    output_dir,
    batch_size,
    output_base_qual,
    bin_size=500,
    adaptive_bin_threshold=10000,
    min_reads_per_bin=0,
    max_padding_fraction=0.20,
    num_workers=4,
):
    """Preprocess reads from a sequence file in parallel using multiple threads."""
    total_files = len(file_list)

    if total_files < num_workers:
        num_workers = total_files
        logger.info(f"Adjusting number of workers to {num_workers} since there are only {total_files} files.")

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for file_path in file_list:
            executor.submit(
                extract_and_bin_reads,
                file_path,
                batch_size,
                output_dir,
                output_base_qual,
                bin_size,
                adaptive_bin_threshold,
            )

    end_time = time.time()
    logger.info(f"Processed {total_files} files in {end_time - start_time:.2f} seconds.")

    os.system("rm " + output_dir + "/*.lock")

    if min_reads_per_bin > 0:
        merge_sparse_bins(
            output_dir,
            min_bin_width=bin_size,
            min_reads_per_bin=min_reads_per_bin,
            max_padding_fraction=max_padding_fraction,
        )

    convert_tsv_to_parquet(output_dir, row_group_size=1000000)


def convert_tsv_to_parquet(tsv_dir, row_group_size=1000000):
    """Convert all TSV files in a directory to Parquet format."""
    logger.info("Converting TSV files to Parquet files...")
    os.makedirs(tsv_dir, exist_ok=True)

    tsv_files = glob.glob(os.path.join(tsv_dir, "*.tsv"))
    read_index = {}

    for tsv_file in tsv_files:
        bin_name = os.path.basename(tsv_file).split(".")[0]
        try:
            # --- sniff header to build an explicit dtype map ---
            with open(tsv_file, "r", encoding="utf-8", errors="ignore") as fh:
                header_cols = fh.readline().rstrip("\n").split("\t")

            dtypes = {
                "ReadName": pl.Utf8,
                "read": pl.Utf8,
                "read_length": pl.Int64,
            }
            if "base_qualities" in header_cols:
                dtypes["base_qualities"] = pl.Utf8

            # read lazily with safe CSV options ---
            df = pl.scan_csv(
                tsv_file,
                separator="\t",
                has_header=True,
                quote_char=None,  # do not treat quotes specially
                encoding="utf8-lossy",  # tolerate odd bytes
                dtypes=dtypes,  # force schema
                infer_schema_length=0,
            )  # don't re-infer

            parquet_file = os.path.join(tsv_dir, f"{bin_name}.parquet")

            logger.info(f"Converting {tsv_file}")
            df.sink_parquet(parquet_file, compression="snappy", row_group_size=row_group_size)
            logger.info(f"Converted {tsv_file} to {parquet_file}")

            # Build the read index efficiently (only pull ReadName)
            names = (
                pl.scan_csv(
                    tsv_file,
                    separator="\t",
                    has_header=True,
                    quote_char=None,
                    encoding="utf8-lossy",
                    dtypes={"ReadName": pl.Utf8},
                    infer_schema_length=0,
                )
                .select("ReadName")
                .collect()
                .get_column("ReadName")
                .to_list()
            )
            for rn in names:
                read_index[rn] = os.path.basename(parquet_file)

            os.remove(tsv_file)
            logger.info(f"Removed original TSV file: {tsv_file}")
            gc.collect()

        except Exception as e:
            logger.error(f"Error converting {tsv_file} to Parquet: {e}")

    if read_index:
        index_parquet_file = os.path.join(tsv_dir, "read_index.parquet")
        from utils import get_version

        index_df = pl.DataFrame([{"ReadName": k, "ParquetFile": v} for k, v in read_index.items()])
        index_df = index_df.with_columns(pl.lit(get_version()).alias("tranquillyzer_version"))
        index_df.write_parquet(index_parquet_file)
        logger.info(f"Index file saved at {index_parquet_file}")


def _count_lines(filepath):
    """Fast line count using buffered binary reads."""
    count = 0
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            count += chunk.count(b"\n")
    return count


def _parse_bin_name(bin_name):
    """Extract (start, end) integers from a bin name like '500_549bp'."""
    parts = bin_name.replace("bp", "").split("_")
    return int(parts[0]), int(parts[1])


def merge_sparse_bins(tsv_dir, min_bin_width=50, min_reads_per_bin=50000, max_padding_fraction=0.20):
    """Merge adjacent sparse length bins to reduce per-bin overhead during annotation.

    Walks bins in ascending order. When a bin has fewer than *min_reads_per_bin*
    reads, it is merged with subsequent adjacent bins until the threshold is met
    or the merged width would exceed ``max(min_bin_width, bin_start * max_padding_fraction)``.

    Operates on TSV files in-place (concatenation + rename) before parquet conversion.
    """
    tsv_files = sorted(
        glob.glob(os.path.join(tsv_dir, "*.tsv")),
        key=lambda f: _parse_bin_name(os.path.basename(f).replace(".tsv", ""))[0],
    )
    if not tsv_files:
        return

    # Collect bin metadata: (start, end, read_count, filepath)
    bin_info = []
    for f in tsv_files:
        bin_name = os.path.basename(f).replace(".tsv", "")
        start, end = _parse_bin_name(bin_name)
        n_reads = max(0, _count_lines(f) - 1)  # subtract header
        bin_info.append((start, end, n_reads, f))

    merged_groups = []  # list of (merged_start, merged_end, total_reads, [filepaths])
    i = 0
    while i < len(bin_info):
        grp_start = bin_info[i][0]
        grp_end = bin_info[i][1]
        grp_count = bin_info[i][2]
        grp_files = [bin_info[i][3]]

        max_width = max(min_bin_width, int(max(grp_start, 1) * max_padding_fraction))

        j = i + 1
        while j < len(bin_info) and grp_count < min_reads_per_bin:
            candidate_end = bin_info[j][1]
            if (candidate_end - grp_start + 1) > max_width:
                break
            grp_end = candidate_end
            grp_count += bin_info[j][2]
            grp_files.append(bin_info[j][3])
            j += 1

        merged_groups.append((grp_start, grp_end, grp_count, grp_files))
        i = j

    # Execute merges
    n_merged = 0
    for grp_start, grp_end, grp_count, grp_files in merged_groups:
        if len(grp_files) == 1:
            # Possibly rename if bin boundaries changed (shouldn't happen for single file)
            continue

        merged_name = f"{grp_start}_{grp_end}bp"
        merged_path = os.path.join(tsv_dir, f"{merged_name}.tsv")
        n_merged += len(grp_files)

        with open(merged_path, "w") as out:
            for k, src in enumerate(grp_files):
                with open(src) as inp:
                    for line_idx, line in enumerate(inp):
                        if k > 0 and line_idx == 0:
                            continue  # skip header of non-first files
                        out.write(line)
                os.remove(src)

    if n_merged:
        orig = len(bin_info)
        final = len(merged_groups)
        logger.info(f"Adaptive bin merging: {orig} bins -> {final} bins ({orig - final} sparse bins merged)")


def find_sequence_files(directory):
    """Recursively find all FASTA/FASTQ files in a directory."""
    extensions = ["*.fa", "*.fasta", "*.fa.gz", "*.fasta.gz", "*.fq", "*.fastq", "*.fq.gz", "*.fastq.gz"]
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(directory, ext)))
    return file_list
