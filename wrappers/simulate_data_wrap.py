import logging

logger = logging.getLogger(__name__)


def simulate_data_wrap(
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
    max_trunc_5p=0,
    max_trunc_3p=0,
    min_spacer=0,
    max_spacer=50,
):
    """Generate synthetic labeled training reads and save to pickle."""
    import os
    import random
    import pickle
    import numpy as np
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from scripts.simulate_training_data import generate_training_reads
    from scripts.trained_models import seq_orders, get_training_structures

    length_range = (min_cDNA, max_cDNA)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(base_dir, "..")

    utils_dir = os.path.join(base_dir, "utils")
    utils_dir = os.path.abspath(utils_dir)

    if training_seq_orders_file is None:
        training_seq_orders_file = f"{utils_dir}/seq_orders.yaml"

    seq_order, sequences, barcodes, UMIs, strand = seq_orders(training_seq_orders_file, model_name)
    training_structs = get_training_structures(training_seq_orders_file, model_name)

    if transcriptome:
        logger.info("Loading transcriptome fasta file")
        transcriptome_records = list(SeqIO.parse(transcriptome, "fasta"))
        logger.info("Transcriptome fasta loaded")
    else:
        logger.info("No transcriptome provided. Will generate random transcripts...")
        transcriptome_records = []
        for i in range(num_reads):
            length = random.randint(min_cDNA, max_cDNA)
            seq_str = "".join(np.random.choice(list("ATCG")) for _ in range(length))
            record = SeqRecord(
                Seq(seq_str), id=f"random_transcript_{i + 1}", description=f"Synthetic transcript {i + 1}"
            )
            transcriptome_records.append(record)
        logger.info(f"Generated {len(transcriptome_records)} synthetic transcripts")

    logger.info("Generating reads")
    reads, labels = generate_training_reads(
        num_reads,
        mismatch_rate,
        insertion_rate,
        deletion_rate,
        polyT_error_rate,
        max_insertions,
        training_structs,
        length_range,
        threads,
        rc,
        transcriptome_records,
        max_trunc_5p,
        max_trunc_3p,
        min_spacer,
        max_spacer,
    )
    logger.info("Finished generating reads")

    os.makedirs(f"{output_dir}/simulated_data", exist_ok=True)

    logger.info("Saving the outputs")
    with open(f"{output_dir}/simulated_data/reads.pkl", "wb") as reads_pkl:
        pickle.dump(reads, reads_pkl)
    with open(f"{output_dir}/simulated_data/labels.pkl", "wb") as labels_pkl:
        pickle.dump(labels, labels_pkl)
    logger.info("Outputs saved")
