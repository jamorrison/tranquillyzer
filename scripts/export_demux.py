import logging

logger = logging.getLogger(__name__)


def load_libs():
    """Lazily import and return libraries needed by the demux export helpers."""
    import os
    import gzip
    import shutil

    import polars as pl

    return (
        os,
        gzip,
        shutil,
        pl,
    )


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


def _write_corrected_demux(rows, output_fmt, demuxed_path, ambiguous_path, strand, barcode_columns,
                           include_barcode_quals=False, include_polya=False, gzipped=False):
    """Write demuxed reads from corrected annotations with full whitelist-style headers.

    Parameters
    ----------
    rows : iterable of dict
        Row dicts from corrected annotations. Accepts polars ``df.iter_rows(named=True)``
        or pandas ``df.to_dict("records")``.

    If ``gzipped=True``, writes gzip-compressed output directly (paths should end in .gz).
    """
    import gzip as gzip_mod

    from scripts.correct_barcodes import reverse_complement

    _open = (lambda p: gzip_mod.open(p, "wt")) if gzipped else (lambda p: open(p, "w"))
    with _open(demuxed_path) as demux_fh, _open(ambiguous_path) as amb_fh:
        for row in rows:
            cell_id = row.get("cell_id", "ambiguous")
            orientation = row.get("orientation", "NA")

            cDNA_start = _parse_first_int(row.get("cDNA_Starts"))
            cDNA_end = _parse_first_int(row.get("cDNA_Ends"))
            read_seq = row.get("read")
            if cDNA_start is None or cDNA_end is None or read_seq is None or cDNA_end <= cDNA_start:
                continue

            read_seq = str(read_seq)
            cDNA_sequence = read_seq[cDNA_start:cDNA_end]

            umi_start = _parse_first_int(row.get("UMI_Starts"))
            umi_end = _parse_first_int(row.get("UMI_Ends"))
            umi_sequence = (
                read_seq[umi_start:umi_end]
                if (umi_start is not None and umi_end is not None and umi_end > umi_start)
                else ""
            )

            # Orientation handling
            needs_rc = (orientation == "-" and strand == "fwd") or (orientation == "+" and strand == "rev")
            if needs_rc:
                cDNA_sequence = reverse_complement(cDNA_sequence)
                umi_sequence = reverse_complement(umi_sequence) if umi_sequence else ""

            # Build barcode header fields
            corrected_parts = []
            for bc in barcode_columns:
                corrected_parts.append(f"{bc}:{row.get(f'corrected_{bc}', 'NMF')}")
            corrected_barcodes_str = ";".join(corrected_parts)

            cell_id_str = str(cell_id) if cell_id != "ambiguous" else "ambiguous"
            umi_name = f"_{umi_sequence}" if umi_sequence else ""
            umi_field = f"|UMI:{umi_sequence}" if umi_sequence else ""

            # PolyA handling
            sequence_out = cDNA_sequence
            polya_start = None
            polya_end = None
            if include_polya:
                polya_start = _parse_first_int(row.get("polyA_Starts") or row.get("polyT_Starts"))
                polya_end = _parse_first_int(row.get("polyA_Ends") or row.get("polyT_Ends"))
                if polya_start is not None and polya_end is not None and polya_end > polya_start:
                    polya_seq = read_seq[polya_start:polya_end]
                    if needs_rc:
                        polya_seq = reverse_complement(polya_seq)
                    sequence_out = cDNA_sequence + polya_seq

            target = amb_fh if cell_id == "ambiguous" else demux_fh
            read_name = row.get("ReadName", "read")

            if output_fmt == "fastq":
                base_q = row.get("base_qualities")
                quality_out = str(base_q)[cDNA_start:cDNA_end] if base_q is not None else ""
                if include_polya and polya_start is not None and polya_end is not None and polya_end > polya_start:
                    polya_qual = str(base_q)[polya_start:polya_end] if base_q is not None else ""
                    quality_out = quality_out + polya_qual
                bq_suffix = ""
                if include_barcode_quals:
                    qual_tokens = []
                    base_q_str = str(base_q) if base_q is not None else ""
                    for bc in barcode_columns:
                        bs = _parse_first_int(row.get(f"{bc}_Starts"))
                        be = _parse_first_int(row.get(f"{bc}_Ends"))
                        if bs is not None and be is not None and base_q_str and be > bs:
                            qual_tokens.append(f"{bc}:{base_q_str[bs:be]}")
                    if umi_start is not None and umi_end is not None and base_q_str and umi_end > umi_start:
                        qual_tokens.append(f"UMI:{base_q_str[umi_start:umi_end]}")
                    if qual_tokens:
                        bq_suffix = f"|BQ:{';'.join(qual_tokens)}"
                header = (
                    f"@{read_name}_{cell_id_str}{umi_name} "
                    f"cell_id:{cell_id_str}|Barcodes:{corrected_barcodes_str}{umi_field}|orientation:{orientation}"
                    f"{bq_suffix}"
                )
                target.write(f"{header}\n{sequence_out}\n+\n{quality_out}\n")
            else:
                header = (
                    f">{read_name}_{cell_id_str}{umi_name} "
                    f"cell_id:{cell_id_str}|Barcodes:{corrected_barcodes_str}{umi_field}|orientation:{orientation}"
                )
                target.write(f"{header}\n{sequence_out}\n")


def _write_corrected_demux_polars(chunk, output_fmt, demuxed_path, ambiguous_path, strand, barcode_columns,
                                   include_barcode_quals=False, include_polya=False, gzipped=False):
    """Fully vectorized demux writer using polars expressions (no Python row loop).

    All string slicing, reverse complement, and header assembly run in polars/Rust.
    Only the final ``to_list()`` + ``writelines()`` touches Python.
    """
    import gzip as gzip_mod

    import polars as pl

    if chunk.height == 0:
        return

    # --- Helper: parse comma-separated coordinate column → first int ---
    def _pc(col_name):
        return (
            pl.col(col_name).cast(pl.Utf8).str.split(",").list.first()
            .str.strip_chars()
            .replace(["", "None", "nan", "NaN"], None)
            .cast(pl.Int64, strict=False)
        )

    # --- Helper: vectorized reverse complement via str ops ---
    def _rc_expr(col):
        """Reverse-complement a string column entirely in polars/Rust."""
        return (
            col.str.reverse()
            .str.replace_all("A", "t", literal=True)
            .str.replace_all("T", "a", literal=True)
            .str.replace_all("C", "g", literal=True)
            .str.replace_all("G", "c", literal=True)
            .str.to_uppercase()
        )

    # --- 1. Parse coordinates ---
    coord_exprs = [_pc("cDNA_Starts").alias("_cs"), _pc("cDNA_Ends").alias("_ce")]
    has_umi = "UMI_Starts" in chunk.columns and "UMI_Ends" in chunk.columns
    if has_umi:
        coord_exprs += [_pc("UMI_Starts").alias("_us"), _pc("UMI_Ends").alias("_ue")]
    if include_polya:
        for col, alias in [("polyA_Starts", "_pas"), ("polyA_Ends", "_pae"),
                           ("polyT_Starts", "_pts"), ("polyT_Ends", "_pte")]:
            if col in chunk.columns:
                coord_exprs.append(_pc(col).alias(alias))
    if include_barcode_quals:
        for bc in barcode_columns:
            for col_sfx, a_sfx in [("_Starts", "_bs"), ("_Ends", "_be")]:
                col = f"{bc}{col_sfx}"
                if col in chunk.columns:
                    coord_exprs.append(_pc(col).alias(f"_{bc}{a_sfx}"))

    chunk = chunk.with_columns(coord_exprs)

    # --- 2. Filter valid cDNA rows ---
    chunk = chunk.filter(
        pl.col("_cs").is_not_null() & pl.col("_ce").is_not_null()
        & (pl.col("_ce") > pl.col("_cs")) & pl.col("read").is_not_null()
    )
    if chunk.height == 0:
        return

    # --- 3. Slice sequences (vectorized str.slice in Rust) ---
    needs_rc = (
        ((pl.col("orientation") == "-") & pl.lit(strand == "fwd"))
        | ((pl.col("orientation") == "+") & pl.lit(strand == "rev"))
    )

    cDNA_raw = pl.col("read").str.slice(pl.col("_cs"), pl.col("_ce") - pl.col("_cs"))
    cDNA_final = pl.when(needs_rc).then(_rc_expr(cDNA_raw)).otherwise(cDNA_raw)
    chunk = chunk.with_columns(cDNA_final.alias("_cDNA"))

    # UMI
    if has_umi:
        umi_valid = pl.col("_us").is_not_null() & pl.col("_ue").is_not_null() & (pl.col("_ue") > pl.col("_us"))
        umi_raw = pl.col("read").str.slice(pl.col("_us"), pl.col("_ue") - pl.col("_us"))
        umi_final = pl.when(umi_valid & needs_rc).then(_rc_expr(umi_raw)).otherwise(
            pl.when(umi_valid).then(umi_raw).otherwise(pl.lit(""))
        )
        chunk = chunk.with_columns(umi_final.alias("_umi"))
    else:
        chunk = chunk.with_columns(pl.lit("").alias("_umi"))

    # PolyA
    if include_polya:
        # Coalesce polyA/polyT coordinates
        pa_s = pl.coalesce([pl.col(c) for c in ["_pas", "_pts"] if c in chunk.columns]) if any(c in chunk.columns for c in ["_pas", "_pts"]) else pl.lit(None, dtype=pl.Int64)
        pa_e = pl.coalesce([pl.col(c) for c in ["_pae", "_pte"] if c in chunk.columns]) if any(c in chunk.columns for c in ["_pae", "_pte"]) else pl.lit(None, dtype=pl.Int64)
        polya_valid = pa_s.is_not_null() & pa_e.is_not_null() & (pa_e > pa_s)
        polya_raw = pl.col("read").str.slice(pa_s, pa_e - pa_s)
        polya_seq = pl.when(polya_valid & needs_rc).then(_rc_expr(polya_raw)).otherwise(
            pl.when(polya_valid).then(polya_raw).otherwise(pl.lit(""))
        )
        chunk = chunk.with_columns(polya_seq.alias("_polya"))
        chunk = chunk.with_columns(pl.concat_str([pl.col("_cDNA"), pl.col("_polya")]).alias("_seq_out"))
    else:
        chunk = chunk.with_columns(pl.col("_cDNA").alias("_seq_out"))

    # --- 4. Build barcode string ---
    bc_parts = []
    for bc in barcode_columns:
        corr_col = f"corrected_{bc}"
        if corr_col in chunk.columns:
            bc_val = pl.col(corr_col).cast(pl.Utf8).fill_null("NMF")
        else:
            bc_val = pl.lit("NMF")
        bc_parts.append(pl.concat_str([pl.lit(f"{bc}:"), bc_val]))
    if len(bc_parts) == 1:
        chunk = chunk.with_columns(bc_parts[0].alias("_bc_str"))
    else:
        chunk = chunk.with_columns(
            pl.concat_str(bc_parts, separator=";").alias("_bc_str")
        )

    # --- 5. Build cell_id string ---
    cell_id_col = pl.col("cell_id").cast(pl.Utf8).fill_null("ambiguous") if "cell_id" in chunk.columns else pl.lit("ambiguous")
    chunk = chunk.with_columns(cell_id_col.alias("_cid"))

    # UMI name/field parts
    has_umi_expr = pl.col("_umi").str.len_chars() > 0
    umi_name = pl.when(has_umi_expr).then(pl.concat_str([pl.lit("_"), pl.col("_umi")])).otherwise(pl.lit(""))
    umi_field = pl.when(has_umi_expr).then(pl.concat_str([pl.lit("|UMI:"), pl.col("_umi")])).otherwise(pl.lit(""))
    chunk = chunk.with_columns(umi_name.alias("_umi_name"), umi_field.alias("_umi_field"))

    read_name_col = pl.col("ReadName").cast(pl.Utf8).fill_null("read") if "ReadName" in chunk.columns else pl.lit("read")
    orientation_col = pl.col("orientation").cast(pl.Utf8).fill_null("NA") if "orientation" in chunk.columns else pl.lit("NA")

    # --- 6. Build records ---
    if output_fmt == "fastq":
        # Quality slicing
        bq_col = "base_qualities"
        if bq_col in chunk.columns:
            cDNA_qual = pl.col(bq_col).cast(pl.Utf8).str.slice(pl.col("_cs"), pl.col("_ce") - pl.col("_cs"))
        else:
            cDNA_qual = pl.lit("")

        if include_polya and "_polya" in chunk.columns:
            # Recompute polyA quality
            if bq_col in chunk.columns:
                pa_s_q = pl.coalesce([pl.col(c) for c in ["_pas", "_pts"] if c in chunk.columns]) if any(c in chunk.columns for c in ["_pas", "_pts"]) else pl.lit(None, dtype=pl.Int64)
                pa_e_q = pl.coalesce([pl.col(c) for c in ["_pae", "_pte"] if c in chunk.columns]) if any(c in chunk.columns for c in ["_pae", "_pte"]) else pl.lit(None, dtype=pl.Int64)
                pq_valid = pa_s_q.is_not_null() & pa_e_q.is_not_null() & (pa_e_q > pa_s_q)
                polya_qual = pl.when(pq_valid).then(
                    pl.col(bq_col).cast(pl.Utf8).str.slice(pa_s_q, pa_e_q - pa_s_q)
                ).otherwise(pl.lit(""))
                qual_out = pl.concat_str([cDNA_qual, polya_qual])
            else:
                qual_out = cDNA_qual
        else:
            qual_out = cDNA_qual
        chunk = chunk.with_columns(qual_out.alias("_qual"))

        # Barcode quals suffix
        if include_barcode_quals and bq_col in chunk.columns:
            bq_parts = []
            for bc in barcode_columns:
                bs_col = f"_{bc}_bs"
                be_col = f"_{bc}_be"
                if bs_col in chunk.columns and be_col in chunk.columns:
                    bc_q_valid = pl.col(bs_col).is_not_null() & pl.col(be_col).is_not_null() & (pl.col(be_col) > pl.col(bs_col))
                    bc_q_slice = pl.when(bc_q_valid).then(
                        pl.concat_str([pl.lit(f"{bc}:"), pl.col(bq_col).cast(pl.Utf8).str.slice(pl.col(bs_col), pl.col(be_col) - pl.col(bs_col))])
                    ).otherwise(pl.lit(None))
                    bq_parts.append(bc_q_slice)
            if has_umi:
                umi_q_valid = pl.col("_us").is_not_null() & pl.col("_ue").is_not_null() & (pl.col("_ue") > pl.col("_us"))
                umi_q_slice = pl.when(umi_q_valid).then(
                    pl.concat_str([pl.lit("UMI:"), pl.col(bq_col).cast(pl.Utf8).str.slice(pl.col("_us"), pl.col("_ue") - pl.col("_us"))])
                ).otherwise(pl.lit(None))
                bq_parts.append(umi_q_slice)
            if bq_parts:
                bq_joined = pl.concat_str([p.fill_null("") for p in bq_parts], separator=";")
                bq_suffix = pl.when(bq_joined.str.len_chars() > 0).then(
                    pl.concat_str([pl.lit("|BQ:"), bq_joined])
                ).otherwise(pl.lit(""))
            else:
                bq_suffix = pl.lit("")
        else:
            bq_suffix = pl.lit("")
        chunk = chunk.with_columns(bq_suffix.alias("_bq_sfx"))

        # FASTQ record: @header\nsequence\n+\nquality\n
        record = pl.concat_str([
            pl.lit("@"), read_name_col, pl.lit("_"), pl.col("_cid"), pl.col("_umi_name"),
            pl.lit(" cell_id:"), pl.col("_cid"),
            pl.lit("|Barcodes:"), pl.col("_bc_str"),
            pl.col("_umi_field"),
            pl.lit("|orientation:"), orientation_col,
            pl.col("_bq_sfx"),
            pl.lit("\n"), pl.col("_seq_out"), pl.lit("\n+\n"), pl.col("_qual"), pl.lit("\n"),
        ])
    else:
        # FASTA record: >header\nsequence\n
        record = pl.concat_str([
            pl.lit(">"), read_name_col, pl.lit("_"), pl.col("_cid"), pl.col("_umi_name"),
            pl.lit(" cell_id:"), pl.col("_cid"),
            pl.lit("|Barcodes:"), pl.col("_bc_str"),
            pl.col("_umi_field"),
            pl.lit("|orientation:"), orientation_col,
            pl.lit("\n"), pl.col("_seq_out"), pl.lit("\n"),
        ])

    chunk = chunk.with_columns(record.alias("_record"))

    # --- 7. Split by cell_id and bulk write ---
    _open = (lambda p: gzip_mod.open(p, "wt")) if gzipped else (lambda p: open(p, "w"))

    demux_records = chunk.filter(pl.col("_cid") != "ambiguous")["_record"].to_list()
    amb_records = chunk.filter(pl.col("_cid") == "ambiguous")["_record"].to_list()

    with _open(demuxed_path) as fh:
        fh.writelines(demux_records)
    with _open(ambiguous_path) as fh:
        fh.writelines(amb_records)


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
