"""
Extract text from a folder of PDF files using PyMuPDF.

Outputs one Markdown-formatted .txt file per PDF into the output directory.
These text files are then used by the agent for synthesis extraction.

Usage:
    python skills/mat-synthesis-extraction/scripts/parse_pdfs.py \\
        --pdf-dir /path/to/pdfs \\
        --output-dir /path/to/output_texts

Requirements:
    - Conda environment: base-agent
    - Required packages: pymupdf
"""

import argparse
import json
import sys
from pathlib import Path

import fitz  # pymupdf


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file using PyMuPDF.

    Concatenates text from all pages, separated by page markers.
    Skips pages with no extractable text (e.g. scanned figures).

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text as a single string.
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(f"--- Page {page_num} ---\n{text}")
    doc.close()
    return "\n\n".join(pages)


def process_pdf_folder(pdf_dir: Path, output_dir: Path) -> list[dict]:
    """
    Process all PDF files in a directory and write extracted text to output.

    Args:
        pdf_dir: Directory containing .pdf files.
        output_dir: Directory to write extracted .txt files.

    Returns:
        List of dicts with keys: pdf_name, txt_path, char_count, status.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}", file=sys.stderr)
        return []

    results = []
    for pdf_path in pdf_files:
        txt_path = output_dir / (pdf_path.stem + ".txt")
        text = extract_text_from_pdf(pdf_path)

        if not text.strip():
            status = "empty"
            print(f"  [WARN] No text extracted from {pdf_path.name}", file=sys.stderr)
        else:
            txt_path.write_text(text, encoding="utf-8")
            status = "ok"
            print(f"  [OK] {pdf_path.name} → {txt_path.name} ({len(text)} chars)")

        results.append(
            {
                "pdf_name": pdf_path.name,
                "txt_path": str(txt_path) if status == "ok" else None,
                "char_count": len(text),
                "status": status,
            }
        )

    summary_path = output_dir / "parse_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")
    print(
        f"Processed {len(results)} PDFs — {sum(r['status'] == 'ok' for r in results)} succeeded."
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from a folder of PDFs using PyMuPDF."
    )
    parser.add_argument(
        "--pdf-dir",
        required=True,
        type=Path,
        help="Directory containing input PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write extracted text files and summary JSON.",
    )
    args = parser.parse_args()

    if not args.pdf_dir.exists():
        print(f"Error: --pdf-dir '{args.pdf_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    process_pdf_folder(args.pdf_dir, args.output_dir)


if __name__ == "__main__":
    main()
