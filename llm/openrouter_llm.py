# llm/openrouter_llm.py
import json
from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_MODEL

class OpenRouterLLM:
    """
    Обертка для OpenRouter API (через openai клиент).
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("❌ API-ключ OpenRouter не найден! Установи его в config.py или через переменную окружения.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.model = model or OPENROUTER_MODEL

    def verify_text(self, image_b64: str, candidate_text: str):
        try:
            content = [
                {
                    "type": "text",
                    "text": (
                        "Ты — помощник OCR. Проверь и исправь распознанный текст "
                        "на русском языке. Верни строго JSON:\n"
                        '{"corrected": "<только сам исправленный текст>", "confidence": <0..1>}.\n'
                        f"Возможный кандидат: {candidate_text}"
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_b64}},
            ]

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise OCR verifier. JSON only."},
                    {"role": "user", "content": content},
                ],
            )

            txt = resp.choices[0].message.content.strip()
            start, end = txt.find("{"), txt.rfind("}")
            if start != -1 and end != -1:
                txt = txt[start:end+1]

            return json.loads(txt)
        except Exception as e:
            return {"corrected": candidate_text, "confidence": 0.0, "error": str(e)}

    def extract_fields(self, text: str, image_b64: str, prompt: str):
        try:
            content = [
                {"type": "text", "text": f"{prompt}\n\n{text}"},
                {"type": "image_url", "image_url": {"url": image_b64}},
            ]

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a contract parser. JSON only."},
                    {"role": "user", "content": content},
                ],
            )

            txt = resp.choices[0].message.content.strip()
            start, end = txt.find("{"), txt.rfind("}")
            if start != -1 and end != -1:
                txt = txt[start:end+1]

            return json.loads(txt)
        except Exception as e:
            return {"error": str(e), "contract_number": "", "date": "", "parties": [], "amount": ""}
