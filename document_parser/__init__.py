"""Document parsing module for PDF files."""

from typing import Any, Dict, List, Callable, Optional
import io
import os
import tempfile

import fitz  # type: ignore
from PIL import Image

from text_recognition import process_image
from text_recognition.utils import pil_to_data_url


BASE_PROMPT = (
	"Ты получаешь распознанный текст документа.\n"
	"Задача: найти ключевые поля договора и вернуть JSON со структурой:\n"
	"{\n"
	'	"№ контракта": {"value": "...", "location": "..."},\n'
	'	"дата заключения": {"value": "...", "location": "..."},\n'
	'	"дата окончания": {"value": "...", "location": "..."},\n'
	'	"контрагент": {"value": ["...", "..."], "location": "..."},\n'
	'	"страна": {"value": "...", "location": "..."},\n'
	'	"сумма контракта": {"value": "...", "location": "..."},\n'
	'	"валюта контракта": {"value": "...", "location": "..."},\n'
	'	"валюта платежа": {"value": "...", "location": "..."}\n'
	"}\n"
	"Каждое поле обязательно к заполнению.\n"
	"Дата должна быть полной (укажи число, месяц и год в жестком формате ДД.ММ.ГГГГ).\n"
	"Возвращай полностью обзац в каждом поле и исправлять слова с ошибками сохраняя структуру текста.\n"
	"В каждом поле JSON обязательно должно быть указано расположение (страница + пункт(арабские цифры) или номер абзаца(абзацы не подписаны цифрой) или номер строки(для заголовков или других)).\n"
	"Использовать переносы текста в ключевых полях можно только если переносы присутствуют на фото, иначе убрать.\n"
)



def parse_document(
    pdf_path: str,
    llm_backend: str = "openrouter",
    log_path: str = "process.log",
    prompt: Optional[str] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
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
    prompt:
        Custom prompt for field extraction.  If ``None`` the built-in
        :data:`BASE_PROMPT` is used.
    progress_cb:
        Optional callback receiving ``(progress, description)`` updates where
        ``progress`` is a float from 0 to 1.
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
        total_pages = len(doc)
        if progress_cb:
            progress_cb(0.0, "Начало")
        for page_index, page in enumerate(doc, start=1):
            log_file.write(f"Processing page {page_index}\n")
            pix = page.get_pixmap()
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name
            ocr_kwargs.pop("use_llm", None)
            info = process_image(
                image_path=tmp_path,
                llm_backend=llm_backend,
                use_llm=False,
                **ocr_kwargs,
            )
            os.unlink(tmp_path)
            text = "\n".join(info.get("verified_lines", []))
            image_b64 = pil_to_data_url(image)
            pages_info.append({"page": page_index, "info": info})
            pages_for_llm.append({"page": page_index, "text": text, "image_b64": image_b64})
            log_file.write(f"Page {page_index}: {len(info.get('verified_lines', []))} lines\n")
            if progress_cb:
                progress_cb(page_index / (total_pages + 1), f"Обработка страницы {page_index}/{total_pages}")

        from llm.router import LLMRouter

        llm = LLMRouter(backend=llm_backend)
        if progress_cb:
            progress_cb(total_pages / (total_pages + 1), "Извлечение полей LLM")
        fields = llm.extract_fields(pages_for_llm, prompt or BASE_PROMPT)
        log_file.write("LLM extraction complete\n")
        if progress_cb:
            progress_cb(1.0, "Готово")

    return {"pages": pages_info, "fields": fields}


__all__ = ["parse_document", "BASE_PROMPT"]
