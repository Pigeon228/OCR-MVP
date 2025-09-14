"""Document parsing module."""

from typing import Any, Dict

from text_recognition import process_image


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

    from llm.router import LLMRouter

    llm = LLMRouter(backend=llm_backend)
    fields = llm.extract_fields(info["blocks"])
    return {"info": info, "fields": fields}


__all__ = ["parse_document"]
