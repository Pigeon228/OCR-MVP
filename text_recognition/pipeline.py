import os
import json
from typing import Any, Dict, Optional

from PIL import Image, ImageDraw, ImageFont

from llm.router import LLMRouter
from .utils import pil_to_data_url, np_convert


def process_image(
    image_path: str,
    output_dir: Optional[str] = None,
    use_llm: bool = False,
    llm_backend: str = "openrouter",
    conf_min: float = 0.1,
    llm_check_max: float = 0.5,
    label_max_chars: int = 30,
    font_size: int = 8,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """Run OCR pipeline with optional LLM verification.

    When ``save_outputs`` is False the function will not write any files to disk
    and instead returns the processed OCR blocks directly.
    """
    import easyocr

    if save_outputs and not output_dir:
        raise ValueError("output_dir must be provided when save_outputs=True")

    overlay_path = (
        os.path.join(output_dir, "easy_overlay.png") if save_outputs else ""
    )
    easy_txt = (
        os.path.join(output_dir, "easy_results.txt") if save_outputs else ""
    )
    verified_txt = (
        os.path.join(output_dir, "verified_results.txt") if save_outputs else ""
    )
    blocks_json = (
        os.path.join(output_dir, "blocks.json") if save_outputs else ""
    )
    crops_dir = os.path.join(output_dir, "crops") if save_outputs else ""

    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(crops_dir, exist_ok=True)

    easy = easyocr.Reader(["ru", "en"], gpu=False)
    results_easy = easy.readtext(image_path, detail=1)
    if not results_easy:
        raise SystemExit("⚠️ EasyOCR не нашёл текста.")

    base = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0)) if save_outputs else None
    draw = ImageDraw.Draw(overlay) if overlay else None

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    llm = LLMRouter(backend=llm_backend) if use_llm else None

    easy_lines, verified_lines, blocks_log = [], [], []
    kept = 0

    for idx, line in enumerate(results_easy, 1):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = line[0]
        easy_text, easy_conf = line[1], float(line[2] or 0.0)

        if not easy_text.strip() or easy_conf < conf_min:
            continue

        xmin, xmax = int(min(x1, x2, x3, x4)), int(max(x1, x2, x3, x4))
        ymin, ymax = int(min(y1, y2, y3, y4)), int(max(y1, y2, y3, y4))
        crop = base.crop((xmin, ymin, xmax, ymax)).convert("RGB")
        crop_path = ""
        if save_outputs:
            crop_path = os.path.join(crops_dir, f"block_{idx}.png")
            crop.save(crop_path)

        final_text, final_conf, source = easy_text, easy_conf, "EASY"
        llm_resp = None

        if use_llm and easy_conf < llm_check_max:
            llm_resp = llm.verify_text(pil_to_data_url(crop), easy_text)
            llm_text = llm_resp.get("corrected", "")
            llm_conf = llm_resp.get("confidence", 0.0)
            if llm_text and llm_conf >= final_conf:
                final_text, final_conf, source = llm_text, llm_conf, "LLM"

        easy_lines.append(easy_text)
        verified_lines.append(final_text)
        kept += 1

        poly = [
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (int(x3), int(y3)),
            (int(x4), int(y4)),
        ]
        color = (0, 0, 255) if source == "EASY" else (255, 128, 0)

        if draw:
            draw.polygon(poly, fill=color + (60,))
            for i in range(4):
                draw.line([poly[i], poly[(i + 1) % 4]], fill=color + (200,), width=2)

            short = (
                final_text[:label_max_chars] + "…"
                if len(final_text) > label_max_chars
                else final_text
            )
            label = f"{source}: {short} ({final_conf:.2f})"
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            draw.rectangle(
                [(poly[0][0], poly[0][1] - th - 4), (poly[0][0] + tw + 4, poly[0][1])],
                fill=(0, 0, 0, 160),
            )
            draw.text(
                (poly[0][0] + 2, poly[0][1] - th - 2),
                label,
                fill=(255, 255, 255, 255),
                font=font,
            )

        blocks_log.append(
            {
                "index": int(idx),
                "coords": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "x3": float(x3),
                    "y3": float(y3),
                    "x4": float(x4),
                    "y4": float(y4),
                },
                "easy": {"text": easy_text, "confidence": float(easy_conf)},
                "llm": llm_resp or {},
                "final": {
                    "text": final_text,
                    "confidence": float(final_conf),
                    "source": source,
                },
                "crop_path": crop_path if save_outputs else None,
            }
        )

    if save_outputs and overlay:
        result_img = Image.alpha_composite(base, overlay).convert("RGB")
        result_img.save(overlay_path)

        with open(easy_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(easy_lines))

        with open(verified_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(verified_lines))

        with open(blocks_json, "w", encoding="utf-8") as f:
            json.dump(blocks_log, f, ensure_ascii=False, indent=2, default=np_convert)

        return {
            "kept": kept,
            "overlay_path": overlay_path,
            "easy_txt": easy_txt,
            "verified_txt": verified_txt,
            "blocks_json": blocks_json,
            "crops_dir": crops_dir,
            "blocks": blocks_log,
        }

    return {"kept": kept, "blocks": blocks_log}
