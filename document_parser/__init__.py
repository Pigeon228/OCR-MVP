"""Document parsing module."""

from typing import Any, Dict

from PIL import Image

from text_recognition import process_image
from text_recognition.utils import pil_to_data_url


def parse_document(image_path: str, llm_backend: str = "openrouter", **ocr_kwargs: Any) -> Dict[str, Any]:
    """Run OCR and extract structured fields using an LLM.

    Parameters
    ----------
    image_path:
        Path to the input image to parse.
    llm_backend:
        Backend name for :class:`llm.router.LLMRouter`.
    **ocr_kwargs:
        Additional keyword arguments forwarded to
        :func:`text_recognition.process_image`.

    Returns
    -------
    dict
        A dictionary containing the OCR ``info`` from ``process_image`` and
        the extracted ``fields`` from the LLM.
    """
    info = process_image(image_path=image_path, llm_backend=llm_backend, **ocr_kwargs)

    full_text = "\n".join(info.get("verified_lines", []))
    image_b64 = pil_to_data_url(Image.open(image_path).convert("RGB"))
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
    fields = llm.extract_fields(full_text, image_b64, prompt)
    return {"info": info, "fields": fields}


__all__ = ["parse_document"]
