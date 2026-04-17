import logging

import numpy as np
from textwrap import fill
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)


def visualize_sequence_annotations(
    colors,
    read_name,
    read,
    predicted_labels,
    architecture,
    reason,
    chars_per_line=100,
    header_max_length=100,
    max_chunks_per_page=50,
):
    """Create a matplotlib figure showing per-position segment annotations for a read."""
    if not read:  # Check for empty reads
        print(f"Warning: Empty read for {read_name}. Skipping this read.")
        return []

    # Ensure read and predicted_labels have the same length
    predicted_labels = predicted_labels[0 : len(read)]
    if len(read) != len(predicted_labels):
        print(f"Error: Length mismatch between read and predicted_labels for {read_name}. Skipping this read.")
        return []

    num_chunks = int(np.ceil(len(read) / chars_per_line))
    read_chunks = [read[i * chars_per_line : (i + 1) * chars_per_line] for i in range(num_chunks)]
    label_chunks = [predicted_labels[i * chars_per_line : (i + 1) * chars_per_line] for i in range(num_chunks)]

    # Determine header content and split it into multiple lines if it exceeds the header_max_length
    header_text = f"{read_name} (Architecture: {architecture}, Reason: {reason})"
    wrapped_header = fill(header_text, width=header_max_length)

    figures = []

    for page_start in range(0, num_chunks, max_chunks_per_page):
        # Select the chunks for the current page
        page_end = min(page_start + max_chunks_per_page, num_chunks)
        page_read_chunks = read_chunks[page_start:page_end]
        page_label_chunks = label_chunks[page_start:page_end]

        # Calculate the figure height dynamically based on the number of lines (chunks)
        fixed_font_size = 10  # Fixed font size for all reads
        chunk_height = 0.6  # Height for each chunk to ensure enough space for text at fixed font size
        header_height = 1.0  # Height for the header
        num_page_chunks = page_end - page_start

        fig_height = header_height + num_page_chunks * chunk_height  # Dynamically adjust the figure height
        fig, axs = plt.subplots(
            num_page_chunks + 1,
            1,
            figsize=(15, fig_height),
            gridspec_kw={"height_ratios": [header_height] + [1] * num_page_chunks},
            dpi=300,
        )

        # Turn off axis for each subplot
        for ax in axs:
            ax.axis("off")

        # Display the header (read name, architecture, reason) at the top
        axs[0].text(
            0.5,
            0.5,
            wrapped_header,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            transform=axs[0].transAxes,
        )

        # Display the read sequence and labels chunk by chunk
        for ax_idx, (read_chunk, label_chunk) in enumerate(zip(page_read_chunks, page_label_chunks), start=1):
            ax = axs[ax_idx]
            start_idx = 0
            current_label = label_chunk[0]
            for idx, (base, label) in enumerate(zip(read_chunk, label_chunk)):
                if label:
                    ax.text(
                        idx / chars_per_line,
                        1,
                        base,
                        ha="center",
                        va="center",
                        color=colors[label],
                        fontsize=fixed_font_size,
                        fontweight="medium",
                    )

                # Handle label positioning and separation
                if current_label != label or idx == len(read_chunk) - 1:  # End of section or end of chunk
                    if current_label:
                        label_position = start_idx / chars_per_line + (idx - start_idx) / (2 * chars_per_line)
                        ax.text(
                            label_position,
                            0.5,
                            current_label,
                            ha="center",
                            va="center",
                            color=colors[current_label],
                            fontsize=fixed_font_size,
                        )

                    start_idx = idx
                    current_label = label

        plt.subplots_adjust(hspace=0.1)  # Adjust space between lines slightly
        plt.tight_layout()
        figures.append(fig)

    return figures


def save_plots_to_pdf(
    sequences, annotated_reads, read_names, filename, colors, chars_per_line=100, max_chunks_per_page=50
):
    """Generate annotation plots for multiple reads and save them to a PDF."""
    with PdfPages(filename) as pdf:
        from utils import get_version

        d = pdf.infodict()
        d["Creator"] = f"tranquillyzer v{get_version()}"
        d["Producer"] = f"tranquillyzer v{get_version()}"
        for sequence, annotated_read, read_name in zip(sequences, annotated_reads, read_names):
            if not sequence:  # Skip if the sequence is empty
                print(f"Warning: Empty sequence for {read_name}. Skipping.")
                continue
            predicted_labels = [""] * len(sequence)
            for label, regions in annotated_read.items():
                if label in ["architecture", "read", "reason", "orientation", "read_length"]:
                    continue
                for start, end in zip(regions["Starts"], regions["Ends"]):
                    if start >= len(sequence) or end > len(sequence):  # Check for out-of-bound regions
                        print(
                            f"Error: Annotation bounds exceed sequence length for {read_name}. Skipping this annotation."
                        )
                        continue
                    predicted_labels[start:end] = [label] * (end - start)

            architecture = annotated_read.get("architecture", "N/A")
            reason = annotated_read.get("reason", "N/A")

            # Visualize and generate figures for each read
            figures = visualize_sequence_annotations(
                colors, read_name, sequence, predicted_labels, architecture, reason, chars_per_line, max_chunks_per_page
            )
            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)
