import os
import logging
import subprocess
import multiprocessing as mp
from collections import defaultdict, deque
from dataclasses import dataclass

import pysam
from rapidfuzz.distance import Hamming, Levenshtein

logger = logging.getLogger(__name__)


def extract_cb_umi(read_name: str):
    """Extract cell barcode and UMI from a BAM alignment's CB and UR tags."""
    parts = read_name.rsplit("_", 2)
    if len(parts) == 3:
        return parts[1], parts[2], parts[0]
    return None, None, read_name


# UMI distance: fast and indel-safe
def _umi_dist(a: str, b: str, max_d: int) -> int:
    """
    Compute edit distance between two UMIs with early cutoff.
    - Uses fast Hamming when lengths match.
    - Falls back to Levenshtein for indels (only needed when max_d >= 2).
    """
    if len(a) == len(b):
        d_h = Hamming.distance(a, b, score_cutoff=max_d + 1)
        if d_h <= max_d:
            return d_h
        if max_d >= 2:
            return Levenshtein.distance(a, b, score_cutoff=max_d + 1)
        return d_h
    return Levenshtein.distance(a, b, score_cutoff=max_d + 1)


# Minimal BK-tree for approximate search
class BKTree:
    __slots__ = ("root", "dist_fn", "max_children")

    @dataclass
    class Node:
        key: str
        children: dict  # distance -> Node

    def __init__(self, dist_fn):
        """Initialize an empty BK-tree with the given distance function."""
        self.root = None
        self.dist_fn = dist_fn
        self.max_children = 0

    def add(self, key: str):
        """Insert a word into the BK-tree."""
        if self.root is None:
            self.root = BKTree.Node(key, {})
            return
        node = self.root
        while True:
            d = self.dist_fn(key, node.key)
            child = node.children.get(d)
            if child is None:
                node.children[d] = BKTree.Node(key, {})
                self.max_children = max(self.max_children, len(node.children))
                return
            node = child

    def query_within(self, key: str, radius: int):
        """Return a key within radius if found (first hit), else None."""
        if self.root is None:
            return None
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = self.dist_fn(key, node.key)
            if d <= radius:
                return node.key
            lo, hi = d - radius, d + radius
            for dd, child in node.children.items():
                if lo <= dd <= hi:
                    stack.append(child)
        return None


class Deduper:
    """
    Streaming deduper that:
      - Buckets by (key, start_bin, end_bin) with bin_size = position_tolerance.
      - Uses a BK-tree per bucket to find an existing UMI within umi_ld.
      - Evicts old bins as we stream forward to keep memory flat.
    Decision is made **only on primary alignments**.
    """

    def __init__(
        self,
        umi_ld: int,
        per_cell: bool,
        stranded: bool,
        position_tolerance: int = 10,
    ):
        """Initialize the deduplicator with a UMI distance threshold and CB tag name."""
        self.umi_ld = umi_ld
        self.per_cell = per_cell
        self.stranded = stranded
        self.bin = max(1, position_tolerance)

        # key -> start_bin -> end_bin -> {'bktree': BKTree, 'kept_umis': set()}
        self.state = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.active_start_bins = defaultdict(deque)  # key -> deque of start_bins in order

        # stats (primary decisions only)
        self.unique_count = 0
        self.dup_count = 0

    def _key_of(self, chrom: str, cb: str, strand: str):
        """Return the deduplication grouping key for an alignment."""
        if self.per_cell:
            return (chrom, cb, strand) if self.stranded else (chrom, cb)
        return (chrom, strand) if self.stranded else (chrom,)

    def _bins(self, start: int, end: int):
        """Return or create the BK-tree bins dict for a grouping key."""
        return start // self.bin, end // self.bin

    def _mk_bktree(self):
        """Create a new BK-tree seeded with the given UMI."""
        def df(a, b, cutoff=self.umi_ld):
            return _umi_dist(a, b, cutoff)

        return BKTree(df)

    def evict_before(self, chrom_key, current_start_bin: int):
        """Remove cached bins for positions before the given reference position."""
        keep_from = max(0, current_start_bin - 2)
        dq = self.active_start_bins[chrom_key]
        while dq and dq[0] < keep_from:
            old_start_bin = dq.popleft()
            self.state[chrom_key].pop(old_start_bin, None)

    def decide_primary(self, chrom: str, cb: str, strand: str, start: int, end: int, umi: str):
        """
        Decide duplicate status for a **primary** alignment.
        Returns (dup_str, canonical_umi):
          - ("No", umi)  for unique reads (keeps its own UMI)
          - ("Yes", hit)  for duplicates (gets the UMI of the kept read)
        """
        chrom_key = self._key_of(chrom, cb, strand)
        sb, eb = self._bins(start, end)

        # Stream forward eviction
        self.evict_before(chrom_key, sb)

        # Ensure bucket exists
        chrom_dict = self.state[chrom_key]
        if sb not in chrom_dict:
            chrom_dict[sb] = defaultdict(dict)
            self.active_start_bins[chrom_key].append(sb)

        end_dict = chrom_dict[sb]
        bucket = end_dict.get(eb)
        if not bucket:
            bucket = {"bktree": self._mk_bktree(), "kept_umis": set()}
            end_dict[eb] = bucket

        # BK-tree query
        hit = bucket["bktree"].query_within(umi, self.umi_ld)
        if hit is None:
            bucket["bktree"].add(umi)
            bucket["kept_umis"].add(umi)
            self.unique_count += 1
            return "No", umi
        else:
            self.dup_count += 1
            return "Yes", hit


# Per-contig worker
def process_region(
    sorted_bam: str,
    temp_bam_path: str,
    region: str,
    umi_ld: int,
    per_cell: bool,
    stranded: bool,
    threads_bgzf: int,
):
    """
    Process a single contig/region:
      - Make duplicate decisions **on primary alignments only**.
      - Propagate decision to secondary/supp if we've already seen the primary in this worker.
      - Write output BAM (coordinate-sorted within this contig).
    """
    deduper = Deduper(
        umi_ld=umi_ld,
        per_cell=per_cell,
        stranded=stranded,
        position_tolerance=10,
    )

    # qname -> (is_dup: bool, canonical_umi: str) for primary decisions within this region
    qname_dup = {}

    with pysam.AlignmentFile(sorted_bam, "rb") as bam_in:
        hdr = bam_in.header.to_dict()
        hdr.setdefault("HD", {})
        hdr["HD"]["SO"] = "coordinate"
        from utils import get_version
        hdr.setdefault("PG", []).append(
            {"ID": "tranquillyzer-dedup", "PN": "tranquillyzer", "VN": get_version()}
        )
        out_header = pysam.AlignmentHeader.from_dict(hdr)

        with pysam.AlignmentFile(temp_bam_path, "wb", header=out_header, threads=threads_bgzf) as bam_out:
            for read in bam_in.fetch(region):
                if read.is_unmapped:
                    continue

                cb, umi, clean_name = extract_cb_umi(read.query_name)
                if not cb or not umi:
                    # No CB/UMI → cannot dedup; write through unchanged
                    bam_out.write(read)
                    continue

                strand = "-" if read.is_reverse else "+"

                # Decide only on PRIMARY
                is_primary = not (read.is_secondary or read.is_supplementary)
                if is_primary:
                    dup_str, canonical_umi = deduper.decide_primary(
                        read.reference_name, cb, strand, read.reference_start, read.reference_end, umi
                    )
                    qname_dup[clean_name] = (dup_str == "Yes", canonical_umi)
                else:
                    # For secondary/supplementary: reuse decision if known
                    cached = qname_dup.get(clean_name)
                    if cached is not None:
                        dup_str = "Yes" if cached[0] else "No"
                        canonical_umi = cached[1]
                    else:
                        dup_str = None
                        canonical_umi = umi

                # Mutate read in-place — preserves all original tags (AS, NM, MD, etc.)
                read.query_name = clean_name

                if dup_str == "Yes":
                    read.flag = read.flag | 0x400
                elif dup_str == "No":
                    read.flag = read.flag & ~0x400

                read.set_tag("CB", cb, value_type="Z")
                read.set_tag("UB", canonical_umi, value_type="Z")
                if dup_str is not None:
                    read.set_tag("DT", dup_str, value_type="Z")

                bam_out.write(read)

    return region, temp_bam_path, deduper.unique_count, deduper.dup_count


# Merge temp BAMs in @SQ order
def merge_in_sq_order(output_bam: str, temp_paths_in_order, template_bam: str, threads_bgzf: int):
    """Merge sorted BAM shards in reference sequence order into a single output via samtools cat."""
    cmd = ["samtools", "cat", "-h", template_bam, "-o", output_bam] + list(temp_paths_in_order)
    subprocess.run(cmd, check=True)


def deduplication_parallel(
    sorted_bam: str,
    output_bam: str,
    umi_ld: int = 1,
    per_cell: bool = True,
    threads: int = max(1, mp.cpu_count() // 2),
    stranded: bool = False,
    bgzf_threads_per_writer: int = 4,
):
    """
    Parallel dedup per contig with per-read semantics:
      * Decide duplicates on **primary** alignments only.
      * Exactly one primary per cluster is unique; the rest are duplicates.
      * Write per-contig BAMs with multithreaded BGZF, then merge in @SQ order.
      * Compute **per-read** stats (primaries only).
    """
    # Discover regions in @SQ order
    with pysam.AlignmentFile(sorted_bam, "rb") as bam:
        regions = list(bam.references)

    temp_paths = {}
    total_unique = 0
    total_dup = 0
    logger.info(f"Splitting per contig across {threads} processes ...")

    # Process per region
    if threads > 1:
        work = []
        with mp.Pool(processes=threads, maxtasksperchild=32) as pool:
            for region in regions:
                temp_path = f"{output_bam}.{region}.tmp.bam"
                work.append(
                    pool.apply_async(
                        process_region,
                        (
                            sorted_bam,
                            temp_path,
                            region,
                            umi_ld,
                            per_cell,
                            stranded,
                            bgzf_threads_per_writer,
                        ),
                    )
                )

            for w in work:
                region, path, n_unique, n_dup = w.get()
                temp_paths[region] = path
                total_unique += n_unique
                total_dup += n_dup
    else:
        # Serial path (no Pool)
        for region in regions:
            temp_path = f"{output_bam}.{region}.tmp.bam"
            region_out, path_out, n_unique, n_dup = process_region(
                sorted_bam,
                temp_path,
                region,
                umi_ld,
                per_cell,
                stranded,
                bgzf_threads_per_writer,
            )
            temp_paths[region_out] = path_out
            total_unique += n_unique
            total_dup += n_dup

    # Merge in @SQ order — use first temp BAM as header source (carries @PG line)
    ordered_paths = [temp_paths[r] for r in regions if r in temp_paths]
    header_bam = ordered_paths[0] if ordered_paths else sorted_bam
    merge_in_sq_order(output_bam, ordered_paths, header_bam, bgzf_threads_per_writer)

    # Cleanup temps
    for p in ordered_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    # Write stats from accumulated counts (no need to re-scan the BAM)
    logger.info("Duplicate marking complete, writing per-read stats")
    stats_file = output_bam.replace(".bam", "_stats.tsv")
    with open(stats_file, "w") as fh:
        fh.write("Metric\tValue\n")
        fh.write(f"Unique Reads\t{total_unique}\n")
        fh.write(f"Duplicate Reads\t{total_dup}\n")
    logger.info("Computed per-read stats, indexing final BAM")
