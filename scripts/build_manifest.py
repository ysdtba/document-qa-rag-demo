"""Create or refresh data/manifest.csv from the PDFs in data/raw/.

File names must look like ``<ticker>_<fiscal_year>_<language>.pdf``,
for example ``0388_2025_en.pdf``.

The script recomputes SHA256 for every PDF and keeps any fields you already
filled in by hand (company, source_url, download_date). New rows get the
file's modification date as download_date; check it and fill company and
source_url yourself.

Run: python scripts/build_manifest.py
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"

FIELDS = ["file_name", "ticker", "company", "fiscal_year", "language", "source_url", "download_date", "sha256"]
NAME_RE = re.compile(r"^(?P<ticker>\d{4,5})_(?P<year>\d{4})_(?P<lang>[a-z]{2})\.pdf$", re.IGNORECASE)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path = MANIFEST) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["file_name"]: row for row in csv.DictReader(fh)}


def main() -> int:
    pdfs = sorted(p for p in RAW_DIR.glob("*.pdf") if p.is_file())
    if not pdfs:
        print(f"No PDFs found in {RAW_DIR}. Download the annual reports first.")
        return 1

    existing = read_manifest()
    rows: list[dict[str, str]] = []
    problems: list[str] = []

    for pdf in pdfs:
        match = NAME_RE.match(pdf.name)
        if not match:
            problems.append(f"{pdf.name}: rename to <ticker>_<year>_<lang>.pdf, e.g. 0388_2025_en.pdf")
            continue
        old = existing.get(pdf.name, {})
        row = {
            "file_name": pdf.name,
            "ticker": match["ticker"],
            "company": old.get("company", ""),
            "fiscal_year": match["year"],
            "language": match["lang"].lower(),
            "source_url": old.get("source_url", ""),
            "download_date": old.get("download_date") or date.fromtimestamp(pdf.stat().st_mtime).isoformat(),
            "sha256": sha256_of(pdf),
        }
        if old.get("sha256") and old["sha256"] != row["sha256"]:
            problems.append(f"{pdf.name}: SHA256 changed since the last manifest; re-check the source")
        for field in ("company", "source_url"):
            if not row[field]:
                problems.append(f"{pdf.name}: fill in {field} in {MANIFEST.relative_to(ROOT)}")
        rows.append(row)

    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {MANIFEST.relative_to(ROOT)}")
    for problem in problems:
        print(f"  ! {problem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
