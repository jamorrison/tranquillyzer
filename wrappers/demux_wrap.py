import os
import gzip
import shutil

import polars as pl


def _load_df(input_file):
    if input_file.endswith(".parquet"):
        return pl.read_parquet(input_file)
    return pl.read_csv(input_file, separator="\t")


def _gzip_file(path):
    if not os.path.exists(path):
        return
    gz_path = path + ".gz"
    if os.path.exists(gz_path):
        os.remove(gz_path)
    with open(path, "rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(path)


def demux_wrap(input_dir, output_dir, input_file, output_fmt):
    input_file = input_file or f"{input_dir}/annotations_valid_bc_corrected.parquet"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Corrected annotation file not found: {input_file}")

    df = _load_df(input_file)
    required = {"demux_bucket", "demux_header", "demux_sequence", "cell_id"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in corrected file: {missing}")

    os.makedirs(output_dir, exist_ok=True)
    out_dir = f"{output_dir}/demuxed_fasta"
    os.makedirs(out_dir, exist_ok=True)

    if output_fmt == "fastq":
        demuxed_path = f"{out_dir}/demuxed.fastq"
        ambiguous_path = f"{out_dir}/ambiguous.fastq"
    else:
        demuxed_path = f"{out_dir}/demuxed.fasta"
        ambiguous_path = f"{out_dir}/ambiguous.fasta"

    with open(demuxed_path, "w") as demux_fh, open(ambiguous_path, "w") as amb_fh:
        for row in df.iter_rows(named=True):
            bucket = str(row["demux_bucket"])
            header = str(row["demux_header"])
            seq = str(row["demux_sequence"])
            target = amb_fh if bucket == "ambiguous" else demux_fh

            if output_fmt == "fastq":
                quality = row.get("demux_quality", None)
                quality = "" if quality is None else str(quality)
                target.write(f"{header}\n{seq}\n+\n{quality}\n")
            else:
                target.write(f"{header}\n{seq}\n")

    _gzip_file(demuxed_path)
    _gzip_file(ambiguous_path)
