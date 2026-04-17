import logging

logger = logging.getLogger(__name__)


def _load_df(input_file):
    """Read an annotation file (Parquet or TSV) into a Polars DataFrame."""
    import polars as pl

    if input_file.endswith(".parquet"):
        return pl.read_parquet(input_file)
    return pl.read_csv(input_file, separator="\t")


def _gzip_file(path):
    """Compress a file in-place with gzip and remove the original."""
    import os
    import gzip
    import shutil

    if not os.path.exists(path):
        return
    gz_path = path + ".gz"
    if os.path.exists(gz_path):
        os.remove(gz_path)
    with open(path, "rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(path)


def _has_usable_demux_qualities(df):
    """Return True if the DataFrame has at least one non-null demux_quality value."""
    if "demux_quality" not in df.columns:
        return False
    for value in df["demux_quality"].to_list():
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"none", "nan"}:
            return True
    return False


def _parse_first_int(value):
    """Extract the first integer from a comma-separated string, or return None."""
    if value is None:
        return None
    token = str(value).split(",")[0].strip()
    if token in {"", "None", "nan", "NaN"}:
        return None
    try:
        return int(float(token))
    except (TypeError, ValueError):
        return None


def _has_usable_base_qualities(df):
    """Return True if the DataFrame has at least one non-null base_qualities value."""
    if "base_qualities" not in df.columns:
        return False
    for value in df["base_qualities"].to_list():
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"none", "nan"}:
            return True
    return False


def _write_from_demux_columns(df, output_fmt, demuxed_path, ambiguous_path):
    """Write demuxed reads to FASTA/FASTQ using pre-computed demux columns."""
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
                if header.startswith("@"):
                    header = ">" + header[1:]
                target.write(f"{header}\n{seq}\n")


def _write_bulk_from_annotations(df, output_fmt, demuxed_path, ambiguous_path):
    """Write bulk cDNA reads to FASTA/FASTQ by slicing from annotation coordinates."""
    with open(demuxed_path, "w") as demux_fh, open(ambiguous_path, "w") as amb_fh:
        _ = amb_fh
        for row in df.iter_rows(named=True):
            cDNA_start = _parse_first_int(row.get("cDNA_Starts"))
            cDNA_end = _parse_first_int(row.get("cDNA_Ends"))
            read = row.get("read")
            if cDNA_start is None or cDNA_end is None or read is None or cDNA_end <= cDNA_start:
                continue
            sequence = str(read)[cDNA_start:cDNA_end]
            read_name = row.get("ReadName", "read")
            orientation = row.get("orientation", "NA")

            if output_fmt == "fastq":
                base_q = row.get("base_qualities")
                quality = str(base_q)[cDNA_start:cDNA_end] if base_q is not None else ""
                if len(quality) < len(sequence):
                    quality = quality + ("!" * (len(sequence) - len(quality)))
                elif len(quality) > len(sequence):
                    quality = quality[: len(sequence)]
                header = f"@{read_name} orientation:{orientation}"
                demux_fh.write(f"{header}\n{sequence}\n+\n{quality}\n")
            else:
                header = f">{read_name} orientation:{orientation}"
                demux_fh.write(f"{header}\n{sequence}\n")
