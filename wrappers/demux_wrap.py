import os
import gzip
import shutil
import logging

import polars as pl

logger = logging.getLogger(__name__)


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


def _has_usable_demux_qualities(df):
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


def demux_wrap(input_dir, output_dir, input_file, output_fmt):
    if input_file is None:
        corrected_default = f"{input_dir}/annotations_valid_bc_corrected.parquet"
        annotation_default = f"{input_dir}/annotations_valid.parquet"
        if os.path.exists(corrected_default):
            input_file = corrected_default
        elif os.path.exists(annotation_default):
            input_file = annotation_default
        else:
            input_file = corrected_default
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Annotation file not found: {input_file}")

    df = _load_df(input_file)
    has_demux_columns = {"demux_bucket", "demux_header", "demux_sequence", "cell_id"}.issubset(set(df.columns))
    has_bulk_columns = {"ReadName", "read", "cDNA_Starts", "cDNA_Ends"}.issubset(set(df.columns))
    if not has_demux_columns and not has_bulk_columns:
        raise ValueError(
            "Input file does not contain demux columns or bulk annotation columns required for FASTA/FASTQ export."
        )

    os.makedirs(output_dir, exist_ok=True)
    out_dir = f"{output_dir}/demuxed_fasta"
    os.makedirs(out_dir, exist_ok=True)

    effective_output_fmt = output_fmt
    if output_fmt == "fastq":
        has_quality = _has_usable_demux_qualities(df) if has_demux_columns else _has_usable_base_qualities(df)
        if not has_quality:
            logger.warning("Base quality scores not available; requested FASTQ demux output will be written as FASTA.")
            effective_output_fmt = "fasta"

    if effective_output_fmt == "fastq":
        demuxed_path = f"{out_dir}/demuxed.fastq"
        ambiguous_path = f"{out_dir}/ambiguous.fastq"
    else:
        demuxed_path = f"{out_dir}/demuxed.fasta"
        ambiguous_path = f"{out_dir}/ambiguous.fasta"

    if has_demux_columns:
        _write_from_demux_columns(df, effective_output_fmt, demuxed_path, ambiguous_path)
    else:
        logger.info("Demux columns not found; exporting bulk reads from annotation coordinates.")
        _write_bulk_from_annotations(df, effective_output_fmt, demuxed_path, ambiguous_path)

    _gzip_file(demuxed_path)
    _gzip_file(ambiguous_path)
