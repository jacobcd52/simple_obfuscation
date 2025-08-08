#!/usr/bin/env python3
"""Build an arXiv ML papers dataset compatible with the PromptBuilder API.

Outputs a directory containing:
- ``manifest.jsonl``: one JSON object per paper with metadata and text path
- ``texts/<arxiv_id>.txt``: extracted plain text from each PDF

Fast path: uses ``gsutil`` to copy PDFs from ``gs://arxiv-dataset`` (Kaggle mirror),
with a progress bar and optional cap on the number of papers to download.
Fallback: can optionally use the arXiv API to fetch PDFs directly if requested.

Filtering: keeps only ML-related categories (``cs.LG``, ``stat.ML``, ``eess.IV``,
``cs.AI`` when intersecting with ML keywords in abstract/title). The category
list is configurable.

Example usage:
  python scripts/build_arxiv_ml_dataset.py \
      --out-dir /data/arxiv-ml \
      --months 2401 2402 2403 \
      --max-papers 500

Requirements:
- ``gsutil`` installed and authenticated if using GCS path
- Python packages: ``pandas``, ``pytz``, ``tqdm``, ``pdfminer.six``
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


# ---------------------------- Config & Helpers ---------------------------- #


DEFAULT_ML_CATEGORIES = {
    # Core ML
    "cs.LG",
    "stat.ML",
    # Related areas often containing ML content
    "cs.AI",
    "cs.CV",
    "cs.CL",
    "cs.IR",
    "cs.NE",
    "eess.IV",
}

ARXIV_GCS_ROOT = "gs://arxiv-dataset/arxiv/arxiv/pdf"


def run_cmd(cmd: Sequence[str]) -> Tuple[int, str, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


def have_gsutil() -> bool:
    code, _, _ = run_cmd(["which", "gsutil"])
    return code == 0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path: Path, txt_path: Path) -> bool:
    _configure_pdfminer_logging()
    from pdfminer.high_level import extract_text

    try:
        text = extract_text(str(pdf_path))
        if not text or not text.strip():
            return False
        txt_path.write_text(text, encoding="utf-8")
        return True
    except Exception:
        return False


_PDFMINER_LOGGING_CONFIGURED = False


def _configure_pdfminer_logging() -> None:
    global _PDFMINER_LOGGING_CONFIGURED
    if _PDFMINER_LOGGING_CONFIGURED:
        return
    # Silence verbose warnings/errors emitted by pdfminer parsing odd PDFs
    for name in ("pdfminer", "pdfminer.pdfinterp", "pdfminer.cmapdb", "pdfminer.layout"):
        logging.getLogger(name).setLevel(logging.ERROR)
    _PDFMINER_LOGGING_CONFIGURED = True


def parse_arxiv_id_from_filename(path: Path) -> Optional[str]:
    # Expect filenames like 2403/2403.01234v2.pdf or 2003/2003.00123.pdf
    name = path.name
    m = re.match(r"(\d{4}\.\d{5})(v\d+)?\.pdf$", name)
    if m:
        return m.group(1)
    return None


def load_or_fetch_metadata_for_ids(arxiv_ids: List[str]) -> List[dict]:
    """Fetch minimal metadata via arXiv API (batch) for given IDs.

    We keep dependencies light by using a tiny Atom feed fetch and regex parse.
    For robustness, if the API fails, we return basic records with placeholders.
    """
    import requests
    from xml.etree import ElementTree as ET

    if not arxiv_ids:
        return []

    # arXiv API supports up to ~50 ids in one call.
    records: List[dict] = []
    batch_size = 50
    for i in range(0, len(arxiv_ids), batch_size):
        batch = arxiv_ids[i : i + batch_size]
        ids_str = ",".join(batch)
        url = f"https://export.arxiv.org/api/query?id_list={ids_str}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
        except Exception:
            for aid in batch:
                records.append(
                    {
                        "arxiv_id": aid,
                        "title": "",
                        "authors": [],
                        "categories": "",
                        "abstract": "",
                    }
                )
            continue

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        for entry in root.findall("atom:entry", ns):
            id_text = entry.findtext("atom:id", default="", namespaces=ns)
            m = re.search(r"arxiv\.org/abs/([\d\.]+)", id_text)
            arxiv_id = m.group(1) if m else ""
            title = entry.findtext("atom:title", default="", namespaces=ns).strip()
            abstract = entry.findtext("atom:summary", default="", namespaces=ns).strip()
            cats = [c.attrib.get("term", "") for c in entry.findall("atom:category", ns)]
            authors = [a.findtext("atom:name", default="", namespaces=ns) for a in entry.findall("atom:author", ns)]
            records.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": [a for a in authors if a],
                    "categories": " ".join([c for c in cats if c]),
                    "abstract": abstract,
                }
            )
    return records


def category_is_ml(categories: str, ml_categories: set[str]) -> bool:
    cat_set = set(categories.split()) if categories else set()
    if cat_set & ml_categories:
        return True
    return False


def gsutil_list_pdfs_for_month(month: str) -> List[str]:
    """List PDF URIs for a given yymm month directory on GCS."""
    src_dir = f"{ARXIV_GCS_ROOT}/{month}"
    code, out, err = run_cmd(["gsutil", "ls", f"{src_dir}/*"])
    if code != 0:
        # Fallback to recursive listing if simple glob fails
        code2, out2, _ = run_cmd(["gsutil", "ls", "-r", src_dir])
        if code2 != 0:
            return []
        lines = [ln.strip() for ln in out2.splitlines() if ln.strip()]
    else:
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return [ln for ln in lines if ln.lower().endswith(".pdf")]


def gsutil_download_pdf_to_dir(uri: str, dest_dir: Path) -> bool:
    """Download a single GCS URI (PDF) into dest_dir. Returns True on success."""
    ensure_dir(dest_dir)
    code, _, _ = run_cmd(["gsutil", "-q", "cp", uri, str(dest_dir)])
    return code == 0


# ----------------------------- Main Workflow ----------------------------- #


def build_dataset(
    out_dir: Path,
    months: List[str],
    *,
    limit: Optional[int],
    max_papers: Optional[int],
    ml_categories: set[str],
    workers: int,
    skip_download: bool,
    use_api_fallback: bool,
) -> None:
    ensure_dir(out_dir)
    texts_dir = out_dir / "texts"
    ensure_dir(texts_dir)
    pdfs_dir = out_dir / "pdfs"
    ensure_dir(pdfs_dir)

    # 1) Download PDFs via gsutil for each requested yymm month directory (with progress)
    if not skip_download and have_gsutil():
        downloaded_ok = 0
        remaining = max_papers if max_papers is not None else None
        for month in months:
            if remaining is not None and remaining <= 0:
                break
            month_uris = gsutil_list_pdfs_for_month(month)
            if not month_uris:
                print(f"Warning: no PDFs listed for month {month} (or listing failed)")
                continue
            if remaining is not None:
                month_uris = month_uris[: remaining]

            dest_dir_for_month = pdfs_dir / month
            ensure_dir(dest_dir_for_month)

            def _dl(uri: str) -> bool:
                return gsutil_download_pdf_to_dir(uri, dest_dir_for_month)

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                for ok in tqdm(ex.map(_dl, month_uris), total=len(month_uris), desc=f"download {month}"):
                    downloaded_ok += 1 if ok else 0

            if remaining is not None:
                remaining -= len(month_uris)
    elif not skip_download:
        print("gsutil not found; skipping GCS download. You can pre-populate the 'pdfs' dir.")

    # 2) Collect candidate PDF files
    pdf_files = sorted([p for p in pdfs_dir.rglob("*.pdf")])
    if limit is not None:
        pdf_files = pdf_files[:limit]

    # 3) Extract arXiv IDs and fetch metadata
    arxiv_ids = []
    id_to_pdf: dict[str, Path] = {}
    for p in pdf_files:
        aid = parse_arxiv_id_from_filename(p)
        if not aid:
            continue
        arxiv_ids.append(aid)
        id_to_pdf[aid] = p

    meta_records = load_or_fetch_metadata_for_ids(arxiv_ids) if arxiv_ids else []
    id_to_meta = {m["arxiv_id"]: m for m in meta_records}

    # 4) Filter to ML categories
    filtered_ids: List[str] = []
    for aid in arxiv_ids:
        cats = id_to_meta.get(aid, {}).get("categories", "")
        if category_is_ml(cats, ml_categories):
            filtered_ids.append(aid)

    if limit is not None:
        filtered_ids = filtered_ids[:limit]

    # 5) Extract text in parallel
    tasks: List[Tuple[str, Path, Path]] = []
    for aid in filtered_ids:
        pdf_path = id_to_pdf[aid]
        txt_path = texts_dir / f"{aid}.txt"
        tasks.append((aid, pdf_path, txt_path))

    def _worker(task: Tuple[str, Path, Path]) -> Tuple[str, bool]:
        aid, pdf_path, txt_path = task
        success = extract_text_from_pdf(pdf_path, txt_path)
        return aid, success

    results: dict[str, bool] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for aid, ok in tqdm(ex.map(_worker, tasks), total=len(tasks), desc="extract"):
            results[aid] = ok

    # 6) Write manifest.jsonl
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        kept = 0
        for aid in filtered_ids:
            if not results.get(aid, False):
                continue
            meta = id_to_meta.get(aid, {})
            record = {
                "arxiv_id": aid,
                "title": meta.get("title", ""),
                "authors": meta.get("authors", []),
                "categories": meta.get("categories", ""),
                "abstract": meta.get("abstract", ""),
                "text_path": str((out_dir / "texts" / f"{aid}.txt").resolve()),
            }
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote manifest with {kept} records: {manifest_path}")
    print("To use with the PromptBuilder: ")
    print("  from src.prompt_builders.arxiv_ml import ArxivMlPromptBuilder")
    print(f"  builder = ArxivMlPromptBuilder(manifest_path='{manifest_path}')")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build arXiv ML dataset for PromptBuilder")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--months",
        nargs="*",
        default=[],
        help="List of yymm directories to download from GCS (e.g., 2403 2404). If empty, assumes PDFs are pre-populated in out-dir/pdfs",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of PDFs to process after download (processing cap)")
    parser.add_argument("--max-papers", type=int, default=None, help="Max number of papers to download across all months (download cap)")
    parser.add_argument(
        "--ml-categories",
        nargs="*",
        default=sorted(list(DEFAULT_ML_CATEGORIES)),
        help="ArXiv category terms considered ML",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel text extraction workers")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip gsutil download; use pre-populated PDFs in out-dir/pdfs",
    )
    parser.add_argument(
        "--api-fallback",
        action="store_true",
        help="Enable arXiv API fallback for metadata (on by default when needed)",
    )

    args = parser.parse_args()

    out_dir: Path = args.out_dir.resolve()
    months: List[str] = list(args.months)
    limit: Optional[int] = args.limit
    workers: int = args.workers
    ml_categories = set(args.ml_categories)
    skip_download: bool = args.skip_download
    use_api_fallback: bool = args.api_fallback or True
    max_papers: Optional[int] = args.max_papers

    build_dataset(
        out_dir=out_dir,
        months=months,
        limit=limit,
        max_papers=max_papers,
        ml_categories=ml_categories,
        workers=workers,
        skip_download=skip_download,
        use_api_fallback=use_api_fallback,
    )


if __name__ == "__main__":
    main()


