"""Verify the downloaded archive, extract it (once) and write data/DATASET_PROVENANCE.json.

Download (not done by this script, the archive is large):
    mkdir -p data/raw && date -u +%Y-%m-%dT%H:%M:%SZ > data/raw/.download_started_utc
    curl -fL -o data/raw/CICIDS2017_improved.zip <SOURCE_URL>
"""
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids.data import ARCHIVE, EXTRACTED, PROVENANCE, RAW, ROOT, SOURCE_URL, sha256  # noqa: E402


def server_headers():
    req = urllib.request.Request(SOURCE_URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return {k: r.headers.get(k) for k in ("Content-Length", "Last-Modified", "ETag", "Content-Type")}


def main():
    with zipfile.ZipFile(ARCHIVE) as z:
        bad = z.testzip()  # checks every member's CRC-32
        if bad:
            sys.exit(f"CRC mismatch in {bad}")
        members = [i for i in z.infolist() if not i.is_dir()]
        if not EXTRACTED.exists():
            EXTRACTED.mkdir(parents=True)
            for i in members:  # flatten, refuse path traversal
                name = Path(i.filename).name
                with z.open(i) as src, open(EXTRACTED / name, "wb") as dst:
                    dst.write(src.read())
                (EXTRACTED / name).chmod(0o444)  # raw files are read-only
    files = []
    for i in members:
        p = EXTRACTED / Path(i.filename).name
        files.append({
            "archive_member": i.filename,
            "extracted_path": str(p.relative_to(ROOT)),
            "compressed_bytes": i.compress_size,
            "uncompressed_bytes": i.file_size,
            "extracted_bytes": p.stat().st_size,
            "zip_crc32": f"{i.CRC:08x}",
            "zip_member_mtime": datetime(*i.date_time).isoformat(),
            "sha256": sha256(p),
        })
    started = RAW / ".download_started_utc"
    prov = {
        "dataset_name": "CIC-IDS2017",
        "variant": "Improved CIC-IDS2017 re-extraction (Liu et al., IEEE CNS 2022)",
        "source_url": SOURCE_URL,
        "documentation_urls": [
            "https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html",
            "https://intrusion-detection.distrinet-research.be/CNS2022/CICIDS2017.html",
            "https://github.com/GintsEngelen/CNS2022_Code",
        ],
        "associated_publication": {
            "citation": "L. Liu, G. Engelen, T. Lynar, D. Essam, W. Joosen. Error Prevalence in NIDS datasets: "
                        "A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018. IEEE CNS 2022, pp. 254-262.",
            "doi": "10.1109/CNS56114.2022.9947235",
        },
        "original_dataset_publication": {
            "citation": "I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani. Toward Generating a New Intrusion Detection "
                        "Dataset and Intrusion Traffic Characterization. ICISSP 2018.",
            "doi": "10.5220/0006639801080116",
        },
        "licence": None,
        "licence_note": "No licence text found on the download or documentation pages; authors request citation.",
        "published_checksum": None,
        "published_checksum_note": "No checksum published on the download page or directory listing (checked 2026-09-23). "
                                   "sha256 below was computed locally after download.",
        "download_started_utc": started.read_text().strip() if started.exists() else None,
        "archive_filename": ARCHIVE.name,
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "archive_zip_crc_check": "passed",
        "server_headers": server_headers(),
        "server_headers_checked_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extracted_files": files,
        "extracted_total_bytes": sum(f["extracted_bytes"] for f in files),
    }
    PROVENANCE.write_text(json.dumps(prov, indent=2) + "\n")
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
