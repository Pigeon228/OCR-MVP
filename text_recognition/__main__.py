"""CLI entry point for the OCR text_recognition module.

This script can be executed either as a module (``python -m text_recognition``) or
directly as a file (``python text_recognition/__main__.py``).  When run directly, the
relative import from ``text_recognition`` would normally fail with ``ImportError``.  To
support both modes, we attempt a relative import first and fall back to an
absolute import after injecting the package's parent directory into
``sys.path`` if necessary.
"""

import argparse
import sys
from pathlib import Path

try:  # pragma: no cover - simple import shim
    from . import process_image
except ImportError:  # running as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from text_recognition import process_image  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Run OCR on an image")
    parser.add_argument(
        "--image",
        default="examples/inputs/contract3.png",
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        default="examples/outputs",
        help="Directory to store results",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM verification",
    )
    parser.add_argument(
        "--backend",
        default="openrouter",
        help="LLM backend name",
    )
    args = parser.parse_args()

    info = process_image(
        image_path=args.image,
        output_dir=args.output,
        use_llm=args.llm,
        llm_backend=args.backend,
    )

    print(f"✅ Принято {info['kept']} блоков")
    print(f"🖼 Оверлей: {info['overlay_path']}")
    print(f"📄 Easy (сырое): {info['easy_txt']}")
    print(f"📄 Итог (после LLM): {info['verified_txt']}")
    print(f"🧾 Лог блоков: {info['blocks_json']}")
    print(f"🖼 Кропы: {info['crops_dir']}")


if __name__ == "__main__":
    main()
