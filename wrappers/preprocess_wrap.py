import logging

logger = logging.getLogger(__name__)


def preprocess_wrap(fasta_dir, output_dir, output_base_qual, chunk_size, bin_size, threads):
    """Discover and preprocess FASTA/FASTQ files into length-binned Parquet."""
    import os
    import time
    import resource

    from scripts.preprocess_reads import (
        parallel_preprocess_data,
        find_sequence_files,
        extract_and_bin_reads,
        convert_tsv_to_parquet,
    )

    os.system("mkdir -p " + output_dir + "/full_length_pp_fa")
    files_to_process = find_sequence_files(fasta_dir)

    start = time.time()
    if len(files_to_process) == 1:
        # If there is only one file, process it in a single thread
        logger.info("Only one file to process. Sorting reads into bins without parallelization.")
        extract_and_bin_reads(files_to_process[0], chunk_size, output_dir + "/full_length_pp_fa", output_base_qual, bin_size)
        os.system(f"rm {output_dir}/full_length_pp_fa/*.lock")

        logger.info("Converting tsv files into parquet")
        convert_tsv_to_parquet(f"{output_dir}/full_length_pp_fa", row_group_size=1000000)
        logger.info("Preprocessing finished!!")
    else:
        # Process files in parallel
        logger.info("Multiple raw files found. Sorting reads into bins with parallelization.")
        parallel_preprocess_data(
            files_to_process, output_dir + "/full_length_pp_fa", chunk_size, output_base_qual, bin_size, num_workers=threads
        )
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    max_rss_mb = usage.ru_maxrss / 1024 if os.uname().sysname == "Linux" else usage.ru_maxrss
    logger.info(f"Peak memory usage: {max_rss_mb:.2f} MB")
    logger.info(f"Elapsed time: {time.time() - start:.2f} seconds")
