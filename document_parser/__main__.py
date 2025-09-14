"""CLI for document_parser."""

import argparse
import json

from . import parse_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract document fields from an image")
    parser.add_argument(
        "--image",
        default="examples/inputs/contract3.png",
        help="Path to the input image",
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

    result = parse_document(
        image_path=args.image,
        llm_backend=args.backend,
        use_llm=args.use_ocr_llm,
    )
    print(json.dumps(result["fields"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
