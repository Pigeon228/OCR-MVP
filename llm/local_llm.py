# llm/local_llm.py
class LocalLLM:
    """
    Заглушка для локальной LLM.
    В будущем можно подключить Ollama, llama.cpp или vLLM.
    """

    def __init__(self, **kwargs):
        pass

    def verify_text(self, image_b64: str, candidate_text: str):
        return {"corrected": candidate_text, "confidence": 0.5, "note": "Local LLM not implemented"}

    def extract_fields(self, pages, prompt: str):
        return {"note": "Local LLM not implemented"}
