"""Offline E012-v2 source/disk gate. Never downloads or starts training.

Budget is conservative capacity planning, not a measured working set.
Downstream corpus preparation is deliberately unavailable while this gate fails.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

RELEASE = "2026_03"
N = 38_840_027
RAW_BYTES = 8_780_552_383
RAW_MD5 = "0492e3cf4093276ae4319ba24d52514f"

def verify_metadata(directory):
    directory = Path(directory)
    tree = ET.parse(directory / "RELEASE.metalink")
    ns = {"m": "http://www.metalinker.org/"}
    if tree.findtext("m:version", namespaces=ns) != RELEASE:
        raise ValueError("wrong release")
    note = (directory / "uniref50.release_note").read_bytes()
    if b"Release: 2026_03" not in note or b"38,840,027" not in note:
        raise ValueError("release note mismatch")
    files = {f.attrib["name"]: f for f in tree.findall("m:files/m:file", ns)}
    fasta = files["uniref50.fasta.gz"]
    if int(fasta.findtext("m:size", namespaces=ns)) != RAW_BYTES:
        raise ValueError("archive size mismatch")
    def md5(element):
        return next(h.text for h in element.findall("m:verification/m:hash", ns)
                    if h.attrib["type"] == "md5")
    if md5(fasta) != RAW_MD5:
        raise ValueError("archive checksum metadata mismatch")
    if hashlib.md5(note).hexdigest() != md5(files["uniref50.release_note"]):
        raise ValueError("release note official checksum mismatch")
    return {"release": RELEASE, "published_raw_entries": N,
            "raw_expected_bytes": RAW_BYTES, "raw_expected_md5": RAW_MD5,
            "release_note_official_md5_verified": True}

def disk_gate(free_bytes):
    # All potentially eligible representatives at the 500-residue contract ceiling.
    # Header/metadata/index allowances are explicit estimates, not server metadata.
    components = {
        "compressed_source": RAW_BYTES,
        "decompressed_source_estimate": 30_000_000_000,
        "filtered_fasta_500_residues_plus_160_header_bytes": N * 660,
        "exact_union_sqlite_sequence_provenance_indexes": (N + 87_930) * 900,
        "protected_clean_fasta_short_ids": (N + 87_930) * 540,
        "mmseqs_databases_indexes_allowance": (N + 87_930) * 700,
        "mmseqs_temporary_allowance_3x_databases": (N + 87_930) * 700 * 3,
        "two_cluster_membership_tables": (N + 87_930) * 48 * 2,
        "final_fasta_manifest_ids_compact_shards": 20_000_000 * 1780,
        "audit_and_transaction_allowance": 1_000_000_000,
    }
    peak = sum(components.values())
    return {"components_bytes": components, "estimated_peak_bytes": peak,
            "preferred_free_bytes_2_5x_peak": (peak * 5 + 1) // 2,
            "available_bytes": free_bytes,
            "peak_capacity_pass": free_bytes >= peak,
            "preferred_margin_pass": free_bytes >= (peak * 5 + 1) // 2,
            "download_authorized_by_gate": free_bytes >= peak,
            "assumptions": "Conservative retained-artifact budget; not measured. "
                "All representatives assumed eligible at max length; actual filtering may reduce it. "
                "Raw decompressed size, metadata and MMseqs temporary sizes are estimates. "
                "Historical immutable data already occupies disk and is not deleted."}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata-dir", type=Path, required=True)
    p.add_argument("--disk", type=Path, default=Path("/mnt/d"))
    a = p.parse_args()
    result = {"source": verify_metadata(a.metadata_dir),
              "disk": disk_gate(shutil.disk_usage(a.disk).free)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["disk"]["peak_capacity_pass"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
