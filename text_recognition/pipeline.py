from typing import Dict, Any, List

from PIL import Image, ImageDraw, ImageFont

from llm.router import LLMRouter
from .utils import pil_to_data_url


def process_image(
    image_path: str,
    use_llm: bool = False,
    llm_backend: str = "openrouter",
    conf_min: float = 0.1,
    llm_check_max: float = 0.5,
    label_max_chars: int = 30,
    font_size: int = 8,
) -> Dict[str, Any]:
    """Run OCR pipeline with optional LLM verification.

    The function performs OCR on ``image_path`` and returns all intermediate
    results without writing anything to disk.  Consumers of this function are
    responsible for persisting any desired outputs.
    """
    import easyocr

    easy = easyocr.Reader(["ru", "en"], gpu=False)
    results_easy = easy.readtext(image_path, detail=1)
    if not results_easy:
        raise SystemExit("⚠️ EasyOCR не нашёл текста.")

    base = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    llm = LLMRouter(backend=llm_backend) if use_llm else None

    easy_lines: List[str] = []
    verified_lines: List[str] = []
    blocks_log: List[Dict[str, Any]] = []
    kept = 0

    for idx, line in enumerate(results_easy, 1):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = line[0]
        easy_text, easy_conf = line[1], float(line[2] or 0.0)

        if not easy_text.strip() or easy_conf < conf_min:
            continue

        xmin, xmax = int(min(x1, x2, x3, x4)), int(max(x1, x2, x3, x4))
        ymin, ymax = int(min(y1, y2, y3, y4)), int(max(y1, y2, y3, y4))
        crop = base.crop((xmin, ymin, xmax, ymax)).convert("RGB")
        crop_data = pil_to_data_url(crop)

        final_text, final_conf, source = easy_text, easy_conf, "EASY"
        llm_resp = None

        if use_llm and easy_conf < llm_check_max:
            llm_resp = llm.verify_text(crop_data, easy_text)
            llm_text = llm_resp.get("corrected", "")
            llm_conf = llm_resp.get("confidence", 0.0)
            if llm_text and llm_conf >= final_conf:
                final_text, final_conf, source = llm_text, llm_conf, "LLM"

        easy_lines.append(easy_text)
        verified_lines.append(final_text)
        kept += 1

        poly = [(int(x1), int(y1)), (int(x2), int(y2)),
                (int(x3), int(y3)), (int(x4), int(y4))]
        color = (0, 0, 255) if source == "EASY" else (255, 128, 0)

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
                "crop_data": crop_data,
            }
        )

    result_img = Image.alpha_composite(base, overlay).convert("RGB")

    return {
        "kept": kept,
        "overlay": result_img,
        "easy_lines": easy_lines,
        "verified_lines": verified_lines,
        "blocks": blocks_log,
    }
