from typing import Optional
import typer
from typing_extensions import Annotated
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")


# Define logging for entire program here
def setup_logger():
    """
    Setup logging for tranquillyzer. Allows for consistent formatting across all
    modules. By putting this in a function, we can also encapsulate all setup in
    one location in case we want to prettify the output. All modules should run

    import logging
    logger = logging.getLogger(__name__)

    at the top of its file to become children of this main logger
    """
    FORMAT = "{asctime} - {name:<35} - {levelname:<7} - {message}"
    logging.basicConfig(format=FORMAT, style="{", level=logging.INFO)

    return logging.getLogger(__name__)


logger = setup_logger()

# =========================
# versioning and app setup
# =========================


try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # type: ignore[assignment]


def get_version() -> str:
    try:
        return version("tranquillyzer")
    except PackageNotFoundError:
        return "unknown"


app = typer.Typer(rich_markup_mode="rich")

# =========================
# shared option help text
# =========================

_HELP_GPU_MEM = (
    "Total GPU memory available, in GB.\n\n"
    "• Single GPU or same-memory GPUs: pass an integer (e.g. [cyan]12[/cyan])\n\n"
    "• Multiple GPUs with different memory: pass a comma-separated list (e.g. [cyan]8,16,32[/cyan])\n\n"
    "• If unset and GPUs are detected, [cyan]12 GB[/cyan] is assumed."
)

_HELP_TARGET_TOKENS = (
    "Approximate token budget [italic]per GPU replica[/italic] used to estimate a safe batch size.\n\n"
    "• One token ≈ one input position after padding (~1 base for DNA).\n\n"
    "• Effective load per replica ≈ batch_size × padded_seq_len.\n\n"
    "• Increase to try larger batches; decrease if you hit OOM.\n\n"
    "• Also guides batch-size heuristics when running on CPU."
)

_HELP_MODEL_NAME = (
    "Base model name — omit any suffix.\n\n"
    "For [bold]CRF[/bold] mode, [cyan]_w_CRF[/cyan] is appended automatically."
)

_HELP_SEQ_ORDER_FILE = (
    "Path to [cyan]seq_orders.tsv[/cyan]. Defaults to the bundled file in [cyan]utils/[/cyan]."
)

_HELP_MODELS_DIR = (
    "Directory containing [cyan]<model>.h5[/cyan] and"
    " [cyan]<model>[_w_CRF]_lbl_bin.pkl[/cyan] files.\n\n"
    "Defaults to the bundled [cyan]models/[/cyan] directory."
)

_HELP_PREPROCESS_DIR = (
    "Directory that holds an existing [cyan]full_length_pp_fa/[/cyan] tree.\n\n"
    "When set, reads are sourced from here instead of [cyan]output_dir[/cyan]."
)

_HELP_MODEL_TYPE_VIZ = (
    "[red]REG[/red] [dim]→[/dim] [green]CNN-LSTM[/green]\n\n"
    "[red]CRF[/red] [dim]→[/dim] [green]CNN-LSTM-CRF[/green]"
)

_HELP_MODEL_TYPE_ANNOT = (
    "[red]REG[/red] [dim]→[/dim] [green]CNN-LSTM[/green]\n\n"
    "[red]CRF[/red] [dim]→[/dim] [green]CNN-LSTM-CRF[/green]\n\n"
    "[red]HYB[/red] [dim]→[/dim] [green]CNN-LSTM[/green] first pass;"
    " [green]CNN-LSTM-CRF[/green] second pass on reads that fail the validity filter"
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"tranquillyzer {get_version()}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Tranquillyzer command-line interface."""
    # Print version banner for all invocations
    typer.echo(f"\nTranquillyzer:v{get_version()}")

    # If invoked without a subcommand → show top-level help
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


# =========================
# available trained models
# =========================


@app.command()
def availableModels():
    """
    Print a catalog of available pretrained models.

    This is a thin CLI wrapper around `scripts.trained_models.trained_models()`.
    It writes model information to stdout for human inspection.

    Usage:
        python main.py availablemodels
    """
    from scripts.trained_models import trained_models

    trained_models()


# ======================================
# available gpus as found by tensorflow
# ======================================


@app.command()
def available_gpus():
    """
    Print GPUs available to tranquillyzer

    Column descriptions:
        Raw Names   - Names as pulled from TensorFlow
        Clean Names - Cleaned names based on number of GPUs
    """
    print("Querying GPUs - this may take some time...")

    import scripts.available_gpus as available_gpus

    available_gpus.available_gpus()


# ===========================================
# extract reads, read_names from fasta file
# ===========================================


@app.command(no_args_is_help=True)
def preprocess(
    fasta_dir: str,
    output_dir: str,
    output_base_qual: bool = typer.Option(False, help="Output base quality scores when available."),
    chunk_size: int = typer.Option(
        100000, help="Base chunk size; effective size scales with the read-length distribution."
    ),
    threads: int = typer.Option(12, help="Number of CPU threads."),
):
    """
    Preprocess raw FASTA/FASTQ files into length-binned Parquet files.

    Steps:
      1) Discover sequence files under `fasta_dir`.
      2) If one file: extract & bin reads serially into TSV chunks,
      then convert to Parquet.
      If multiple: run `parallel_preprocess_data` across `threads`.
      3) Record peak memory and runtime.

    Args:
        fasta_dir: Directory containing input FASTA/FASTQ (optionally gzipped) files.
        output_dir: Directory where outputs are written: `<output_dir>/full_length_pp_fa`.
        output_base_qual: If True, include base qualities in outputs when available.
        chunk_size: Base chunk size for binning; effective size scales with read-length distribution.
        threads: Number of CPU workers for parallel preprocessing.

    Outputs:
        - `<output_dir>/full_length_pp_fa/*.tsv[.lock]` (intermediate, cleaned up)
        - `<output_dir>/full_length_pp_fa/*.parquet` (final)
        - `<output_dir>/full_length_pp_fa/read_index.parquet` (read → parquet file mapping)
        - Logs to stdout/stderr.

    Raises:
        FileNotFoundError: If `fasta_dir` has no readable sequence files.
        RuntimeError: Propagated from subprocesses if failures occur.
    """
    from wrappers.preprocess_wrap import preprocess_wrap

    preprocess_wrap(fasta_dir, output_dir, output_base_qual, chunk_size, threads)


# ==============================
# plot read length distribution
# ==============================


@app.command(no_args_is_help=True)
def readlengthDist(output_dir: str):
    """
    Generate a read-length distribution plot from preprocessed parquet files.

    Reads the binned Parquet files in `<output_dir>/full_length_pp_fa` and
    writes a PNG/PDF plot into `<output_dir>/plots`.

    Args:
        output_dir: Base output directory from the `preprocess` step.

    Outputs:
        `<output_dir>/plots/read_length_distribution.*`

    Raises:
        FileNotFoundError: If required Parquet files are missing.
    """
    from wrappers.read_length_distr_wrap import read_length_distr_wrap

    read_length_distr_wrap(output_dir)


# ===========================================
# inspect selected reads for annotations
# ===========================================


@app.command(no_args_is_help=True)
def visualize(
    output_dir: str,
    preprocess_dir: str = typer.Option(None, "--preprocess-dir", help=_HELP_PREPROCESS_DIR),
    output_file: str = typer.Option(
        "full_read_annots",
        help="Output PDF base name (extension [cyan].pdf[/cyan] is added automatically).",
    ),
    model_name: str = typer.Option("10x3p_sc_ont_011", help=_HELP_MODEL_NAME),
    model_type: Annotated[str, typer.Option(help=_HELP_MODEL_TYPE_VIZ)] = "CRF",
    seq_order_file: str = typer.Option(None, help=_HELP_SEQ_ORDER_FILE),
    models_dir: str = typer.Option(None, "--models-dir", help=_HELP_MODELS_DIR),
    gpu_mem: Annotated[str, typer.Option(help=_HELP_GPU_MEM)] = None,
    target_tokens: Annotated[int, typer.Option(help=_HELP_TARGET_TOKENS)] = 1_200_000,
    vram_headroom: float = typer.Option(0.35, help="Fraction of GPU memory to reserve as headroom."),
    min_batch_size: int = typer.Option(1, help="Minimum batch size for model inference."),
    max_batch_size: int = typer.Option(2000, help="Maximum batch size for model inference."),
    num_reads: int = typer.Option(None, help="Randomly sample this many reads to visualize."),
    read_names: str = typer.Option(None, help="Comma-separated read names to visualize."),
    threads: int = typer.Option(2, help="Number of CPU threads."),
):
    """
    Run model inference on selected reads and export per-read annotation plots.

    You can either provide `read_names` (comma-separated list) to visualize
    specific reads, or supply `num_reads` to randomly sample reads from the
    global `read_index.parquet`.

    Args:
        output_dir: Pipeline base output directory (expects `full_length_pp_fa/`).
        output_file: Base name for the output PDF under `<output_dir>/plots`.
        model_name: Base model name; `_w_CRF` suffix is inferred for CRF mode.
        model_type: One of {"REG","CRF"}; model choice for inference.
        gpu_mem: Optional GPU memory budget string (e.g., "12" or "8,16").
        target_tokens: Token budget per replica to estimate batch sizing.
        vram_headroom: Fraction of VRAM to reserve to reduce OOM risk.
        min_batch_size: Lower bound for batch size search.
        max_batch_size: Upper bound for batch size search.
        num_reads: If provided, randomly sample this many reads to visualize.
        read_names: If provided, comma-separated explicit read IDs to visualize.
        threads: CPU workers for downstream decoding/extraction.

    Outputs:
        `<output_dir>/plots/<output_file>.pdf` — tiled per-read annotation plots.

    Raises:
        ValueError: If neither `num_reads` nor `read_names` is supplied.
        FileNotFoundError: If required indices/files are missing.
    """
    from wrappers.visualize_wrap import visualize_wrap

    visualize_wrap(
        output_dir,
        output_file,
        model_name,
        model_type,
        seq_order_file,
        models_dir,
        gpu_mem,
        target_tokens,
        vram_headroom,
        min_batch_size,
        max_batch_size,
        num_reads,
        read_names,
        threads,
        preprocess_dir=preprocess_dir,
    )


# =========================
# Annotate all the reads
# =========================


@app.command(no_args_is_help=True)
def annotate_reads(
    output_dir: str,
    preprocess_dir: str = typer.Option(None, "--preprocess-dir", help=_HELP_PREPROCESS_DIR),
    whitelist_file: str = typer.Option(
        None, "--whitelist-file", help="Barcode whitelist TSV. Required for barcode correction/demux."
    ),
    model_name: str = typer.Option("10x3p_sc_ont_011", help=_HELP_MODEL_NAME),
    model_type: Annotated[str, typer.Option(help=_HELP_MODEL_TYPE_ANNOT)] = "HYB",
    seq_order_file: str = typer.Option(None, help=_HELP_SEQ_ORDER_FILE),
    models_dir: str = typer.Option(None, "--models-dir", help=_HELP_MODELS_DIR),
    chunk_size: int = typer.Option(
        100000, help="Base chunk size; effective size scales with the read-length distribution."
    ),
    combine_chunk_outputs: bool = typer.Option(
        True,
        help=(
            "Merge all chunk TSV outputs into a single"
            " [cyan]annotations_valid/invalid.parquet[/cyan].\n\n"
            "Disable to keep per-chunk parquet outputs."
        ),
    ),
    keep_chunk_tsv_after_combine: bool = typer.Option(
        False,
        help=(
            "Keep chunk TSV files after successful combine.\n\n"
            "By default they are deleted when [cyan]--combine-chunk-outputs[/cyan] is enabled."
        ),
    ),
    keep_demux_chunk_outputs_after_combine: bool = typer.Option(
        False,
        "--keep-demux-chunk-after-combine",
        help=(
            "Keep demux chunk FASTA/FASTQ files after successful combine.\n\n"
            "By default they are deleted when [cyan]--run-demux[/cyan] is enabled."
        ),
    ),
    checkpoint_file: str = typer.Option(
        None,
        help="Checkpoint file path. Defaults to [cyan]<output_dir>/annotation_checkpoint.txt[/cyan].",
    ),
    resume: bool = typer.Option(
        True,
        help=(
            "Resume from checkpoint and chunk markers when available.\n\n"
            "Disable to clear prior annotate-reads outputs and re-run from the start."
        ),
    ),
    gpu_mem: Annotated[str, typer.Option(help=_HELP_GPU_MEM)] = None,
    target_tokens: Annotated[int, typer.Option(help=_HELP_TARGET_TOKENS)] = 1_200_000,
    vram_headroom: float = typer.Option(0.35, help="Fraction of GPU memory to reserve as headroom."),
    min_batch_size: int = typer.Option(1, help="Minimum batch size for model inference."),
    max_batch_size: int = typer.Option(8192, help="Maximum batch size for model inference."),
    bc_lv_threshold: int = typer.Option(2, help="Levenshtein-distance threshold for barcode correction."),
    threads: int = typer.Option(12, help="Number of CPU threads for barcode correction and demultiplexing."),
    max_queue_size: int = typer.Option(3, help="Max number of Parquet files queued for post-processing."),
    include_polya: bool = typer.Option(
        False, help="Append detected polyA tails to output sequences (includes qualities in FASTQ)."
    ),
    run_barcode_correction: bool = typer.Option(
        False, help="Run barcode correction on valid annotated reads. Disabled by default."
    ),
    include_barcode_quals: bool = typer.Option(
        False,
        help=(
            "Append base qualities for barcode segments\n\n"
            "into the FASTQ header when writing FASTQ."
        ),
    ),
    run_demux: bool = typer.Option(
        False,
        help=(
            "Write FASTA/FASTQ during annotation.\n\n"
            "With [cyan]--run-barcode-correction[/cyan], reads are demultiplexed by cell;\n\n"
            "otherwise valid reads are exported in bulk."
        ),
    ),
    output_fmt: str = typer.Option(
        "fasta",
        "--demux-output-fmt",
        "--output-fmt",
        help=(
            "Output format for demultiplexed reads\n\n"
            "when [cyan]--run-demux[/cyan] is enabled: [cyan]fasta[/cyan] or [cyan]fastq[/cyan]."
        ),
    ),
):
    """
    Annotation-first pipeline with optional barcode correction and demultiplexing.

    Pipeline:
      1) Iterate through binned Parquet files and run model predictions.
      2) Post-process predictions to call segment boundaries and write per-bin/per-chunk outputs.
      3) Optionally run barcode correction and demultiplexing inline in the same pass.
      4) Optionally (HYB) re-run invalid reads with CRF model.
      5) Finalize chunk outputs to combined Parquet or per-chunk Parquet outputs.

    Args:
        output_dir: Base directory with `full_length_pp_fa/` and target for outputs.
        whitelist_file: TSV with valid barcode columns; required only when barcode correction is enabled.
        output_fmt: "fasta" or "fastq" for demultiplexed outputs.
        model_name: Base model label (without `_w_CRF`).
        model_type: "REG", "CRF", or "HYB" (REG pass then CRF on invalid).
        chunk_size: Row group size for final Parquet conversions.
        gpu_mem: Optional GPU memory budget string (e.g., "12" or "8,16").
        target_tokens: Token budget per replica to guide batching.
        vram_headroom: Fraction of VRAM to keep free as safety margin.
        min_batch_size: Min batch size for inference.
        max_batch_size: Max batch size for inference.
        bc_lv_threshold: Levenshtein threshold for barcode correction.
        threads: CPU workers for annotation post-processing.
        max_queue_size: Max in-flight Parquet chunks for worker queueing.
        run_barcode_correction: If true, computes corrected barcode columns and demux stats.
        run_demux: If true, writes `demuxed_fasta/*` files in the same pass (demuxed with barcode correction, bulk otherwise).
        checkpoint_file: Path to checkpoint file storing pass/bin/chunk progress.
        resume: If true, restart from checkpoint and skip done chunks.
        combine_chunk_outputs: If true, merge chunk TSVs into annotations_valid/invalid.parquet.
        keep_chunk_tsv_after_combine: If true with combine enabled, keep chunk TSVs and also write per-chunk parquet.
        keep_demux_chunk_outputs_after_combine: If true with demux enabled, keep chunk FASTA/FASTQ after combine.

    Outputs:
        - Chunk outputs under `<output_dir>/annotation_chunks/`
        - Combined `<output_dir>/annotations_valid.parquet` and `_invalid.parquet` (default)
          OR per-chunk parquet files if `--no-combine-chunk-outputs`
        - Optional: `<output_dir>/demuxed_fasta/demuxed.(fa|fq)` and `ambiguous.(fa|fq)`

    Raises:
        FileNotFoundError: If expected input files or whitelist are missing.
        TimeoutError: If worker result collection stalls beyond threshold.
        RuntimeError: Propagated exceptions from worker processes.
    """
    from wrappers.annotate_reads_wrap import annotate_reads_wrap

    if run_barcode_correction and not whitelist_file:
        raise typer.BadParameter("whitelist_file is required when --run-barcode-correction is enabled")
    if output_fmt not in {"fasta", "fastq"}:
        raise typer.BadParameter("demux output format must be either 'fasta' or 'fastq'")

    annotate_reads_wrap(
        output_dir,
        whitelist_file,
        output_fmt,
        model_name,
        model_type,
        seq_order_file,
        chunk_size,
        gpu_mem,
        target_tokens,
        vram_headroom,
        min_batch_size,
        max_batch_size,
        bc_lv_threshold,
        threads,
        max_queue_size,
        include_barcode_quals,
        include_polya,
        run_barcode_correction,
        run_demux,
        checkpoint_file,
        resume,
        combine_chunk_outputs,
        keep_chunk_tsv_after_combine,
        keep_demux_chunk_outputs_after_combine,
        models_dir=models_dir,
        preprocess_dir=preprocess_dir,
    )


@app.command(no_args_is_help=True)
def barcode_correct(
    input_dir: str,
    whitelist_file: str,
    output_dir: str = typer.Option(None, help="Output directory. Defaults to [cyan]input_dir[/cyan]."),
    input_file: str = typer.Option(
        None,
        help=(
            "Annotations file. Defaults to\n\n"
            " [cyan]<input_dir>/annotations_valid.parquet[/cyan],\n\n"
            " or falls back to [cyan]<input_dir>/annotations_valid_bc_corrected.parquet[/cyan]."
        ),
    ),
    output_fmt: str = typer.Option(
        "fasta",
        help=(
            "Output format for demultiplexed reads\n\n"
            "when [cyan]--run-demux[/cyan] is enabled: [cyan]fasta[/cyan] or [cyan]fastq[/cyan]."
        ),
    ),
    seq_order_file: str = typer.Option(None, help=_HELP_SEQ_ORDER_FILE),
    model_name: str = typer.Option("10x3p_sc_ont_011", help="Model name for seq-order lookup when seq_order_file is set."),
    bc_lv_threshold: int = typer.Option(2, help="Levenshtein-distance threshold for barcode correction."),
    threads: int = typer.Option(12, help="Number of CPU threads for barcode correction."),
    chunk_size: int = typer.Option(100000, help="Number of rows to scan/process per chunk from annotations input."),
    include_barcode_quals: bool = typer.Option(
        False, help="Append barcode qualities to the FASTQ header when writing FASTQ demux output."
    ),
    include_polya: bool = typer.Option(False, help="Append detected polyA tails to demuxed sequences."),
    run_demux: bool = typer.Option(
        False,
        help="Run demuxing concurrently while correcting barcodes (single pass through annotations).",
    ),
    keep_demux_chunk_outputs_after_combine: bool = typer.Option(
        False,
        "--keep-demux-chunk-after-combine",
        help=(
            "Keep demux chunk FASTA/FASTQ files after successful combine.\n\n"
            "By default they are deleted when [cyan]--run-demux[/cyan] is enabled."
        ),
    ),
):
    """
    Correct barcode segments on annotated valid reads, with optional concurrent demultiplexing.
    """
    from wrappers.barcode_correction_wrap import barcode_correction_wrap

    barcode_correction_wrap(
        input_dir=input_dir,
        whitelist_file=whitelist_file,
        output_dir=output_dir or input_dir,
        input_file=input_file,
        output_fmt=output_fmt,
        seq_order_file=seq_order_file,
        model_name=model_name,
        bc_lv_threshold=bc_lv_threshold,
        threads=threads,
        chunk_size=chunk_size,
        include_barcode_quals=include_barcode_quals,
        include_polya=include_polya,
        run_demux=run_demux,
        keep_demux_chunk_outputs_after_combine=keep_demux_chunk_outputs_after_combine,
    )


@app.command(no_args_is_help=True)
def demux_reads(
    input_dir: str,
    output_dir: str = typer.Option(None, help="Output directory. Defaults to [cyan]input_dir[/cyan]."),
    input_file: str = typer.Option(
        None,
        help=(
            "Annotation file to export reads from. Defaults to\n\n"
            " [cyan]<input_dir>/annotations_valid_bc_corrected.parquet[/cyan],\n\n"
            " or falls back to [cyan]<input_dir>/annotations_valid.parquet[/cyan] for bulk export."
        ),
    ),
    output_fmt: str = typer.Option("fasta", help="Output format for demultiplexed reads: [cyan]fasta[/cyan] or [cyan]fastq[/cyan]."),
):
    """
    Write FASTA/FASTQ from annotations.

    If demux columns exist (barcode-corrected file), writes demuxed and ambiguous outputs.
    Otherwise, exports bulk reads from cDNA coordinates in annotations_valid.
    """
    from wrappers.demux_wrap import demux_wrap

    demux_wrap(
        input_dir=input_dir,
        output_dir=output_dir or input_dir,
        input_file=input_file,
        output_fmt=output_fmt,
    )


# ======================================
# QC metrics
# ======================================


@app.command(no_args_is_help=True)
def qc_metrics(
    input_dir: str,
    output_dir: str = typer.Option(
        None,
        help="Output directory for QC files. Defaults to [cyan]<input_dir>/qc_metrics/[/cyan].",
    ),
    valid_file: str = typer.Option(
        None,
        help=(
            "Path to valid-reads parquet.\n\n"
            "Defaults to [cyan]annotations_valid_bc_corrected.parquet[/cyan]"
            " or [cyan]annotations_valid.parquet[/cyan] inside [cyan]input_dir[/cyan]."
        ),
    ),
    invalid_file: str = typer.Option(
        None,
        help=(
            "Path to invalid-reads parquet.\n\n"
            "Defaults to [cyan]annotations_invalid.parquet[/cyan] inside [cyan]input_dir[/cyan]."
        ),
    ),
    sample_name: str = typer.Option(
        None,
        help=(
            "Sample label used as a prefix in all output file names and as the MultiQC sample name.\n\n"
            "Defaults to the base name of [cyan]input_dir[/cyan]."
        ),
    ),
    read_len_bin_width: int = typer.Option(
        100,
        help="Bin width (bp) for the read-length distribution plots.",
    ),
):
    """
    Generate QC metrics from tranquillyzer annotation parquet files.

    Produces MultiQC-compatible ``*_mqc.tsv`` files plus a plain summary TSV.

    Metrics computed:
      1) Total / valid / invalid read counts and rates.
      2) Demuxed and ambiguous read counts (when barcode-corrected parquet is used).
      3) Minimum edit-distance frequency tables for every barcode type present.
      4) Read-length distributions as line plots — all, valid, invalid, demuxed,
         ambiguous.
      5) Cell-barcode knee plot: per-cell read count (log rank vs log count) and
         cumulative fraction of reads.
      6) Per-cell read count table.

    MultiQC usage:
      Run ``multiqc <output_dir>`` (or the parent directory) after this command
      to pick up all ``*_mqc.tsv`` files automatically.
    """
    from wrappers.qc_metrics_wrap import qc_metrics_wrap

    resolved_output = output_dir or os.path.join(input_dir, "qc_metrics")
    resolved_sample = sample_name or os.path.basename(os.path.abspath(input_dir))

    qc_metrics_wrap(
        input_dir=input_dir,
        output_dir=resolved_output,
        valid_file=valid_file,
        invalid_file=invalid_file,
        sample_name=resolved_sample,
        read_len_bin_width=read_len_bin_width,
    )


# ======================================
# align inserts to the reference genome
# ======================================


@app.command(no_args_is_help=True)
def align(
    input_dir: str,
    ref: str,
    output_dir: str,
    preset: str = typer.Option("splice", help="minimap2 preset."),
    filt_flag: str = typer.Option(
        "260",
        help=(
            "Samtools [cyan]-F[/cyan] flag to filter reads by bitwise flag.\n\n"
            " Default [cyan]260[/cyan] drops secondary alignments and unmapped reads."
        ),
    ),
    mapq: int = typer.Option(0, help="Minimum MAPQ for alignments to be included downstream."),
    threads: int = typer.Option(12, help="Number of CPU threads."),
    add_minimap_args: str = typer.Option("", help="Additional minimap2 arguments, appended verbatim."),
):
    """
    Align demultiplexed reads to a reference with minimap2 and index the BAM.

    This command finds `<input_dir>/demuxed_fasta/demuxed.(fastq|fasta)`,
    aligns using `minimap2 -ax <preset>`, filters with samtools (`-F filt_flag -q mapq`),
    sorts and writes a coordinate-sorted BAM, then creates a BAM index.

    Args:
        input_dir: Directory containing `demuxed_fasta` outputs.s
        ref: Reference genome/transcriptome FASTA for minimap2.
        output_dir: Base directory to write `aligned_files/demuxed_aligned.bam`.
        preset: Minimap2 preset (e.g., "splice" for long RNA).
        filt_flag: Samtools view `-F` flag to drop reads by bitwise flags.
        mapq: Minimum MAPQ to keep alignments.
        threads: Number of CPU threads for minimap2/samtools.
        add_minimap_args: Extra args appended to the minimap2 command.

    Outputs:
        `<output_dir>/aligned_files/demuxed_aligned.bam[.bai]`

    Raises:
        FileNotFoundError: If demultiplexed FASTA/FASTQ cannot be located.
        CalledProcessError: If minimap2/samtools commands fail.
    """
    from wrappers.align_wrap import align_wrap

    align_wrap(input_dir, ref, output_dir, preset, filt_flag, mapq, threads, add_minimap_args)


# ==============================
# Deduplication using UMI-tools
# ==============================


@app.command(no_args_is_help=True)
def dedup(
    input_dir: str,
    lv_threshold: int = typer.Option(2, help="Levenshtein-distance threshold for UMI similarity."),
    stranded: bool = typer.Option(True, help="Whether the library is directional (stranded)."),
    per_cell: bool = typer.Option(True, help="Perform UMI correction on a per-cell basis."),
    threads: int = typer.Option(12, help="Number of CPU threads."),
):
    """
    Mark/remove PCR duplicates using UMI-aware clustering on aligned reads.

    Invokes the project-specific `deduplication_parallel` which clusters UMIs
    by Levenshtein distance (threshold `lv_threshold`), optionally per-cell and
    respecting library strandedness.

    Args:
        input_dir: Base directory with `aligned_files/demuxed_aligned.bam`.
        lv_threshold: Edit distance threshold for grouping similar UMIs.
        stranded: Whether library is strand-aware.
        per_cell: If True, perform UMI correction per cell ID.
        threads: CPU threads used by the deduplication pipeline.

    Outputs:
        `<input_dir>/aligned_files/demuxed_aligned_dup_marked.bam[.bai]`

    Raises:
        FileNotFoundError: If expected output BAM is not produced.
        RuntimeError: If underlying tools throw errors.
    """
    from wrappers.dedup_wrap import dedup_wrap

    dedup_wrap(input_dir, lv_threshold, stranded, per_cell, threads)


# ==============================
# Split bam per cell
# ==============================


@app.command(no_args_is_help=True)
def split_bam(
    input_bam: str,
    out_dir: Optional[str] = typer.Option(
        None,
        help="Output directory for per-cell BAMs. Defaults to [cyan]<input_bam_dir>/split_bams[/cyan].",
    ),
    bucket_threads: Optional[int] = typer.Option(
        1,
        help="Worker processes for Stage 1 (per-contig bucketing). Default: all CPUs, capped by contig count.",
    ),
    merge_threads: Optional[int] = typer.Option(
        1,
        help="Worker processes for Stage 2 (per-bucket merge/split). Default: ≤8 (I/O-heavy stage).",
    ),
    nbuckets: int = typer.Option(
        256,
        help="Hash buckets used to partition cell barcodes. \n\n"
        "Higher → fewer CBs per bucket but more temp files. Typical: 128/256/512.",
    ),
    tag: str = typer.Option(
        "CB",
        help="BAM tag that holds the cell barcode. Reads missing this tag are skipped.",
    ),
    max_open_cb_writers: int = typer.Option(
        128,
        help="Max per-CB output BAM writers kept open per process (LRU cache).\n\n"
        "Prevents 'too many open files'.",
    ),
    filter_secondary: bool = typer.Option(False, help="Drop secondary alignments."),
    filter_supplementary: bool = typer.Option(False, help="Drop supplementary alignments."),
    filter_unmapped: bool = typer.Option(True, help="Drop unmapped reads."),
    filter_duplicates: bool = typer.Option(True, help="Drop PCR/optical duplicates."),
    min_mapq: Optional[int] = typer.Option(0, help="Minimum MAPQ to retain an alignment."),
    keep_tmp: bool = typer.Option(False, help="Keep temporary bucket BAM parts (useful for debugging)."),
    index_outputs: bool = typer.Option(False, help="Create a [cyan].bai[/cyan] index for each per-CB BAM."),
    prefer_csi_index: bool = typer.Option(
        False, help="Prefer a CSI index over BAI for the (possibly sorted) input BAM when indexing is needed."
    ),
):
    """
    Split a coordinate-sorted BAM into one BAM per cell barcode tag (default: CB).

    Uses a scalable two-stage strategy:
      1) Parallel by contig: write reads into a fixed number of hash buckets.
      2) Parallel by bucket: write final per-CB BAMs with bounded open file handles.

    Output directory defaults to <input_bam_dir>/split_bams if --out-dir is not provided.
    """
    from wrappers.split_bam_wrap import split_bam_wrap

    if not os.path.exists(input_bam):
        raise typer.BadParameter(f"Input BAM not found: {input_bam}")

    if out_dir is None:
        in_dir = os.path.dirname(os.path.abspath(input_bam)) or "."
        out_dir = os.path.join(in_dir, "split_bams")

    os.makedirs(out_dir, exist_ok=True)

    split_bam_wrap(
        input_bam=input_bam,
        out_dir=out_dir,
        bucket_threads=bucket_threads,
        merge_threads=merge_threads,
        nbuckets=nbuckets,
        tag=tag,
        max_open_cb_writers=max_open_cb_writers,
        filter_secondary=filter_secondary,
        filter_supplementary=filter_supplementary,
        filter_unmapped=filter_unmapped,
        filter_duplicates=filter_duplicates,
        min_mapq=min_mapq,
        keep_tmp=keep_tmp,
        index_outputs=index_outputs,
        prefer_csi_index=prefer_csi_index,
    )


# ===========================
# Simulate training dataset
# ===========================


@app.command(no_args_is_help=True)
def simulate_data(
    model_name: str,
    output_dir: str,
    training_seq_orders_file: str = typer.Option(None, help=_HELP_SEQ_ORDER_FILE),
    num_reads: int = typer.Option(50000, help="Number of reads to simulate."),
    mismatch_rate: float = typer.Option(0.05, help="Base substitution probability."),
    insertion_rate: float = typer.Option(0.05, help="Base insertion probability."),
    deletion_rate: float = typer.Option(0.06, help="Base deletion probability."),
    min_cDNA: int = typer.Option(100, help="Minimum cDNA segment length."),
    max_cDNA: int = typer.Option(500, help="Maximum cDNA segment length."),
    polyT_error_rate: float = typer.Option(0.02, help="Error rate within polyT/polyA segments."),
    max_insertions: float = typer.Option(1, help="Maximum insertions allowed after a single base."),
    threads: int = typer.Option(2, help="Number of CPU threads."),
    rc: bool = typer.Option(
        True,
        help=(
            "Include reverse complements in the training data."
            " The final dataset will contain twice the requested number of reads."
        ),
    ),
    transcriptome: str = typer.Option(None, help="Transcriptome FASTA. If omitted, random transcripts are generated."),
    invalid_fraction: float = typer.Option(0.3, help="Fraction of reads to synthesize as invalid."),
):
    """
    Generate synthetic labeled reads for training, and serialize to PKL.

    The generator uses a model-specific segment order/pattern and either
    provided transcripts or synthetic random transcripts to create realistic
    reads with configurable error profiles.

    Args:
        model_name: Model key for segment order specification.
        output_dir: Destination for `simulated_data/reads.pkl` and `labels.pkl`.
        num_reads: Number of primary reads to synthesize (before RC doubling).
        mismatch_rate: Base substitution probability.
        insertion_rate: Base insertion probability.
        deletion_rate: Base deletion probability.
        min_cDNA: Minimum transcript (cDNA) segment length.
        max_cDNA: Maximum transcript (cDNA) segment length.
        polyT_error_rate: Error rate within polyT/polyA regions.
        max_insertions: Maximum insertions per base position.
        threads: CPU threads used within the generator.
        rc: If True, include reverse complements (doubling dataset size).
        transcriptome: Optional FASTA of transcripts; otherwise random.
        invalid_fraction: Fraction of reads to synthesize as invalid.

    Outputs:
        `<output_dir>/simulated_data/reads.pkl`
        `<output_dir>/simulated_data/labels.pkl`
    """
    from wrappers.simulate_data_wrap import simulate_data_wrap

    simulate_data_wrap(
        model_name,
        output_dir,
        training_seq_orders_file,
        num_reads,
        mismatch_rate,
        insertion_rate,
        deletion_rate,
        min_cDNA,
        max_cDNA,
        polyT_error_rate,
        max_insertions,
        threads,
        rc,
        transcriptome,
        invalid_fraction,
    )


# ===============
#  Train model
# ===============


@app.command(no_args_is_help=True)
def train_model(
    model_name: str,
    output_dir: str,
    param_file: str = typer.Option(
        None, help="Path to [cyan]training_params.tsv[/cyan]. Defaults to the bundled file in [cyan]utils/[/cyan]."
    ),
    training_seq_orders_file: str = typer.Option(None, help=_HELP_SEQ_ORDER_FILE),
    num_val_reads: int = typer.Option(20, help="Number of validation reads to synthesize."),
    mismatch_rate: float = typer.Option(0.05, help="Base substitution probability."),
    insertion_rate: float = typer.Option(0.05, help="Base insertion probability."),
    deletion_rate: float = typer.Option(0.06, help="Base deletion probability."),
    min_cDNA: int = typer.Option(100, help="Minimum cDNA segment length."),
    max_cDNA: int = typer.Option(500, help="Maximum cDNA segment length."),
    polyT_error_rate: float = typer.Option(0.02, help="Error rate within polyT/polyA segments."),
    max_insertions: float = typer.Option(2, help="Maximum insertions allowed after a single base."),
    threads: int = typer.Option(2, help="Number of CPU threads."),
    rc: bool = typer.Option(
        True,
        help=(
            "Include reverse complements in the validation data."
            " The final dataset will contain twice the requested number of reads."
        ),
    ),
    transcriptome: str = typer.Option(None, help="Transcriptome FASTA. If omitted, random transcripts are generated."),
    invalid_fraction: float = typer.Option(0.3, help="Fraction of reads to synthesize as invalid."),
    gpu_mem: Annotated[str, typer.Option(help=_HELP_GPU_MEM)] = None,
    target_tokens: Annotated[int, typer.Option(help=_HELP_TARGET_TOKENS)] = 1_200_000,
    vram_headroom: float = typer.Option(0.35, help="Fraction of GPU memory to reserve as headroom."),
    min_batch_size: int = typer.Option(1, help="Minimum batch size for model inference."),
    max_batch_size: int = typer.Option(2000, help="Maximum batch size for model inference."),
):
    """
    Grid-train model variants from parameter table and export artifacts.

    For a given `model_name`, this reads `utils/training_params.tsv`,
    enumerates parameter combinations, trains each variant using a distributed
    strategy when available, and saves:
      - model weights / SavedModel
      - fitted label binarizer
      - training history
      - validation visualization PDF on a small synthetic set

    Args:
        model_name: Column in `training_params.tsv` whose parameter grid to use.
        output_dir: Base directory to write per-variant subfolders and artifacts.
        num_val_reads: Number of validation reads to synthesize.
        mismatch_rate / insertion_rate / deletion_rate: Error model for validation set.
        min_cDNA / max_cDNA: Transcript length bounds for validation set.
        polyT_error_rate: Error rate inside polyT/polyA for validation set.
        max_insertions: Max insertions per position for validation set.
        threads: CPU threads for validation read synthesis.
        rc: Include reverse complements for validation set.
        transcriptome: Optional validation FASTA; else random transcripts.
        invalid_fraction: Fraction of invalid reads to generate for validation set.
        gpu_mem / target_tokens / vram_headroom / min_batch_size / max_batch_size:
            Parameters to guide validation inference batch sizing (plot stage).

    Outputs (per variant `<model_name>_<idx>`):
        - `*.h5` or SavedModel
        - `<model_name>_<idx>_lbl_bin.pkl`
        - `<model_name>_<idx>_history.tsv`
        - `<model_name>_<idx>_val_viz.pdf`

    Raises:
        FileNotFoundError: If `training_params.tsv` missing or `model_name` not present.
        RuntimeError: If training fails.
    """
    from wrappers.train_model_wrap import train_model_wrap

    train_model_wrap(
        model_name,
        output_dir,
        param_file,
        training_seq_orders_file,
        num_val_reads,
        mismatch_rate,
        insertion_rate,
        deletion_rate,
        min_cDNA,
        max_cDNA,
        polyT_error_rate,
        max_insertions,
        threads,
        rc,
        transcriptome,
        invalid_fraction,
        gpu_mem,
        target_tokens,
        vram_headroom,
        min_batch_size,
        max_batch_size,
    )

    train_model_wrap(
        model_name,
        output_dir,
        param_file,
        training_seq_orders_file,
        num_val_reads,
        mismatch_rate,
        insertion_rate,
        deletion_rate,
        min_cDNA,
        max_cDNA,
        polyT_error_rate,
        max_insertions,
        threads,
        rc,
        transcriptome,
        invalid_fraction,
        gpu_mem,
        target_tokens,
        vram_headroom,
        min_batch_size,
        max_batch_size,
    )


if __name__ == "__main__":
    app()
