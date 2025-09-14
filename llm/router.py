# llm/router.py
from typing import Dict, Any
from .openrouter_llm import OpenRouterLLM
from .local_llm import LocalLLM

class LLMRouter:
    """
    Универсальный роутер для работы с LLM.
    Поддерживает разные бэкенды (локальные и облачные).
    """

    def __init__(self, backend: str = "openrouter", **kwargs):
        self.backend_name = backend.lower()

        if self.backend_name == "openrouter":
            self.backend = OpenRouterLLM(**kwargs)
        elif self.backend_name == "local":
            self.backend = LocalLLM(**kwargs)
        else:
            raise ValueError(f"Неизвестный backend LLM: {backend}")

    def verify_text(self, image_b64: str, candidate_text: str) -> Dict[str, Any]:
        """
        Проверка OCR результата через выбранную LLM.
        Возвращает словарь {"corrected": str, "confidence": float}
        """
        return self.backend.verify_text(image_b64, candidate_text)

    def extract_fields(self, text: str, image_b64: str, prompt: str) -> Dict[str, Any]:
        """
        Выделение ключевых полей из документа.

        Parameters
        ----------
        text:
            Полный извлечённый текст документа.
        image_b64:
            Изображение документа в base64.
        prompt:
            Подсказка для LLM.
        """
        return self.backend.extract_fields(text, image_b64, prompt)
