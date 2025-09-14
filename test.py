import os
import io
import json
import base64
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from llm.router import LLMRouter  # <-- используем наш роутер

# ==== Параметры ====
IMAGE_PATH   = "examples/inputs/contract3.png"
OUTPUT_DIR   = "examples/outputs"
OVERLAY_PATH = os.path.join(OUTPUT_DIR, "easy_overlay.png")
EASY_TXT     = os.path.join(OUTPUT_DIR, "easy_results.txt")
VERIFIED_TXT = os.path.join(OUTPUT_DIR, "verified_results.txt")
BLOCKS_JSON  = os.path.join(OUTPUT_DIR, "blocks.json")
CROPS_DIR    = os.path.join(OUTPUT_DIR, "crops")

CONF_MIN        = 0.1   # минимальная уверенность
LLM_CHECK_MAX   = 0.5   # если conf < этого порога — проверка через LLM
LABEL_MAX_CHARS = 30    # длина подписи
FONT_SIZE       = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

# ==== EasyOCR ====
easy = easyocr.Reader(['ru', 'en'], gpu=False)
results_easy = easy.readtext(IMAGE_PATH, detail=1)
if not results_easy:
    raise SystemExit("⚠️ EasyOCR не нашёл текста.")

# ==== Подготовка рисунка ====
base = Image.open(IMAGE_PATH).convert("RGBA")
overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

try:
    font = ImageFont.truetype("arial.ttf", FONT_SIZE)
except OSError:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

# ==== LLM Router ====
llm = LLMRouter(backend="openrouter")  # всегда один ключ для OpenRouter

# ==== Утилита ====
def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# ==== Обработка ====
easy_lines, verified_lines, blocks_log = [], [], []
kept = 0

for idx, line in enumerate(results_easy, 1):
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = line[0]
    easy_text, easy_conf = line[1], float(line[2] or 0.0)

    if not easy_text.strip() or easy_conf < CONF_MIN:
        continue

    # Кроп блока
    xmin, xmax = int(min(x1, x2, x3, x4)), int(max(x1, x2, x3, x4))
    ymin, ymax = int(min(y1, y2, y3, y4)), int(max(y1, y2, y3, y4))
    crop = base.crop((xmin, ymin, xmax, ymax)).convert("RGB")
    crop_path = os.path.join(CROPS_DIR, f"block_{idx}.png")
    crop.save(crop_path)

    final_text, final_conf, source = easy_text, easy_conf, "EASY"
    llm_resp = None

    # Проверка LLM если OCR неуверен
    if easy_conf < LLM_CHECK_MAX:
        llm_resp = llm.verify_text(pil_to_data_url(crop), easy_text)
        llm_text = llm_resp.get("corrected", "")
        llm_conf = llm_resp.get("confidence", 0.0)
        if llm_text and llm_conf >= final_conf:
            final_text, final_conf, source = llm_text, llm_conf, "LLM"

    easy_lines.append(easy_text)
    verified_lines.append(final_text)
    kept += 1

    # Отрисовка блока
    poly = [(int(x1), int(y1)), (int(x2), int(y2)),
            (int(x3), int(y3)), (int(x4), int(y4))]
    color = (0, 0, 255) if source == "EASY" else (255, 128, 0)

    draw.polygon(poly, fill=color + (60,))
    for i in range(4):
        draw.line([poly[i], poly[(i+1) % 4]], fill=color + (200,), width=2)

    short = (final_text[:LABEL_MAX_CHARS] + "…") if len(final_text) > LABEL_MAX_CHARS else final_text
    label = f"{source}: {short} ({final_conf:.2f})"
    tw, th = draw.textbbox((0, 0), label, font=font)[2:]
    draw.rectangle([(poly[0][0], poly[0][1] - th - 4),
                    (poly[0][0] + tw + 4, poly[0][1])],
                   fill=(0, 0, 0, 160))
    draw.text((poly[0][0] + 2, poly[0][1] - th - 2),
              label, fill=(255, 255, 255, 255), font=font)

    blocks_log.append({
        "index": int(idx),
        "coords": {"x1": float(x1), "y1": float(y1), "x2": float(x2),
                   "y2": float(y2), "x3": float(x3), "y3": float(y3),
                   "x4": float(x4), "y4": float(y4)},
        "easy": {"text": easy_text, "confidence": float(easy_conf)},
        "llm": llm_resp or {},
        "final": {"text": final_text, "confidence": float(final_conf), "source": source},
        "crop_path": crop_path
    })

# ==== Сохранение ====
result_img = Image.alpha_composite(base, overlay).convert("RGB")
result_img.save(OVERLAY_PATH)

with open(EASY_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(easy_lines))

with open(VERIFIED_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(verified_lines))

def np_convert(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)

with open(BLOCKS_JSON, "w", encoding="utf-8") as f:
    json.dump(blocks_log, f, ensure_ascii=False, indent=2, default=np_convert)

print(f"✅ Принято {kept} блоков (min conf ≥ {CONF_MIN})")
print(f"🖼 Оверлей: {OVERLAY_PATH}")
print(f"📄 Easy (сырое): {EASY_TXT}")
print(f"📄 Итог (после LLM): {VERIFIED_TXT}")
print(f"🧾 Лог блоков: {BLOCKS_JSON}")
print(f"🖼 Кропы: {CROPS_DIR}")
