from typing import Any, Dict

from llm.router import LLMRouter
from text_recognition import process_image


def extract_key_lines(image_path: str, llm_backend: str = "openrouter") -> Dict[str, Any]:
    """Extract key fields from a document image.

    The function runs OCR via ``text_recognition`` without writing any files and
    then asks the selected LLM backend to find contract fields, returning them
    as JSON-compatible dictionary.
    """
    ocr_info = process_image(
        image_path=image_path,
        output_dir=None,
        use_llm=False,
        save_outputs=False,
    )
    blocks = ocr_info["blocks"]
    router = LLMRouter(backend=llm_backend)
    return router.extract_fields(blocks)
