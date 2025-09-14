"""Command line interface for :mod:`document_parser`.

This module can be executed either via ``python -m document_parser`` or by
running the file directly.  When executed as a script the package is not
installed, so we fall back to adjusting ``sys.path`` and performing an absolute
import to avoid ``ImportError: attempted relative import with no known parent``.
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract document fields from a PDF")
    parser.add_argument(
        "--pdf",
        default="test.pdf",
        help="Path to the input PDF file",
    )
    parser.add_argument(
        "--backend",
        default="openrouter",
        help="LLM backend name",
    )
    parser.add_argument(
        "--use-ocr-llm",
        action="store_true",
        help="Verify OCR blocks with LLM before field extraction",
    )
    args = parser.parse_args()

    try:  # pragma: no cover - import shim for direct execution
        from . import parse_document
    except ImportError:  # executed as a standalone script
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from document_parser import parse_document  # type: ignore

    result = parse_document(
        pdf_path=args.pdf,
        llm_backend=args.backend,
        use_llm=args.use_ocr_llm,
    )
    print(json.dumps(result["fields"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
