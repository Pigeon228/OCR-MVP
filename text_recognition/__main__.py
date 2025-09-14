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
import os
import json
import base64
import io
from pathlib import Path

from PIL import Image

try:  # pragma: no cover - simple import shim
    from . import process_image
except ImportError:  # running as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from text_recognition import process_image  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Run OCR on an image")
    parser.add_argument(
        "--image",
        default="contract3.png",
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        default="examples/outputs",
        help="Directory to store results",
    )
    parser.add_argument(
        "--backend",
        default="openrouter",
        help="LLM backend name",
    )
    args = parser.parse_args()

    info = process_image(
        image_path=args.image,
        use_llm=False,
        llm_backend=args.backend,
    )

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    overlay_path = os.path.join(output_dir, "easy_overlay.png")
    easy_txt = os.path.join(output_dir, "easy_results.txt")
    verified_txt = os.path.join(output_dir, "verified_results.txt")
    blocks_json = os.path.join(output_dir, "blocks.json")
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    info["overlay"].save(overlay_path)
    with open(easy_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(info["easy_lines"]))
    with open(verified_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(info["verified_lines"]))
    with open(blocks_json, "w", encoding="utf-8") as f:
        json.dump(info["blocks"], f, ensure_ascii=False, indent=2)

    for block in info["blocks"]:
        data = block["crop_data"].split(",", 1)[1]
        Image.open(io.BytesIO(base64.b64decode(data))).save(
            os.path.join(crops_dir, f"block_{block['index']}.png")
        )

    print(f"✅ Принято {info['kept']} блоков")
    print(f"🖼 Оверлей: {overlay_path}")
    print(f"📄 Easy (сырое): {easy_txt}")
    print(f"📄 Итог: {verified_txt}")
    print(f"🧾 Лог блоков: {blocks_json}")
    print(f"🖼 Кропы: {crops_dir}")


if __name__ == "__main__":
    main()
