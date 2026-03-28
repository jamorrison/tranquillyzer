from typing import Optional


def split_bam_wrap(
    input_bam: str,
    out_dir: str,
    bucket_threads: Optional[int] = None,
    merge_threads: Optional[int] = None,
    nbuckets: int = 256,
    tag: str = "CB",
    max_open_cb_writers: int = 128,
    filter_secondary: bool = False,
    filter_supplementary: bool = False,
    filter_unmapped: bool = True,
    filter_duplicates: bool = True,
    min_mapq: Optional[int] = None,
    keep_tmp: bool = False,
    index_outputs: bool = True,
    prefer_csi_index: bool = False,
):
    """Split a BAM file into per-cell-barcode BAM files."""
    from scripts.split_bam_file import split_bam_file

    split_bam_file(
        input_bam,
        out_dir,
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
