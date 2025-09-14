"""Document parsing module for PDF files."""

from typing import Any, Dict, List
import io
import os
import tempfile

import fitz  # type: ignore
from PIL import Image

from text_recognition import process_image
from text_recognition.utils import pil_to_data_url


def parse_document(
    pdf_path: str,
    llm_backend: str = "openrouter",
    log_path: str = "process.log",
    **ocr_kwargs: Any,
) -> Dict[str, Any]:
    """Run OCR on each PDF page and extract structured fields using an LLM.

    Parameters
    ----------
    pdf_path:
        Path to the input PDF to parse.
    llm_backend:
        Backend name for :class:`llm.router.LLMRouter`.
    log_path:
        Where to save the processing log.
    **ocr_kwargs:
        Additional keyword arguments forwarded to
        :func:`text_recognition.process_image`.

    Returns
    -------
    dict
        A dictionary containing ``pages`` with OCR info for each page and the
        extracted ``fields`` from the LLM.
    """
    pages_info: List[Dict[str, Any]] = []
    pages_for_llm: List[Dict[str, str]] = []

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"PDF: {pdf_path}\n")
        doc = fitz.open(pdf_path)
        for page_index, page in enumerate(doc, start=1):
            log_file.write(f"Processing page {page_index}\n")
            pix = page.get_pixmap()
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name
            info = process_image(image_path=tmp_path, llm_backend=llm_backend, **ocr_kwargs)
            os.unlink(tmp_path)
            text = "\n".join(info.get("verified_lines", []))
            image_b64 = pil_to_data_url(image)
            pages_info.append({"page": page_index, "info": info})
            pages_for_llm.append({"page": page_index, "text": text, "image_b64": image_b64})
            log_file.write(f"Page {page_index}: {len(info.get('verified_lines', []))} lines\n")

        prompt = (
            "Ты получаешь распознанный текст документа.\n"
            "Задача: найти ключевые поля договора и вернуть JSON со структурой:\n"
            "{\n"
            '  "contract_number": "...",\n'
            '  "date": "...",\n'
            '  "parties": ["...", "..."],\n'
            '  "amount": "...",\n'
            '  "other": "..." (если есть)\n'
            "}\n"
            "Если поле не найдено — ставь пустую строку.\n"
            "Дата должна быть полной (укажи число, месяц и год в жестком формате ДД.ММ.ГГГГ)."
            "Возвращай полностью обзац + пункт(если есть) в каждом поле и передавать в оригинальном виде."
            "Использовать переносы текста в ключевых полях можно только если переносы присутствуют на фото, иначе убрать."
        )

        from llm.router import LLMRouter

        llm = LLMRouter(backend=llm_backend)
        fields = llm.extract_fields(pages_for_llm, prompt)
        log_file.write("LLM extraction complete\n")

    return {"pages": pages_info, "fields": fields}


__all__ = ["parse_document"]
