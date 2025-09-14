import io
import os
import tempfile
from typing import Dict, Any

import pandas as pd
import streamlit as st

from document_parser import parse_document, BASE_PROMPT

# Настройки страницы
st.set_page_config(page_title="OCR Demo")
st.title("OCR-MVP Demo")

# Загрузка файла
uploaded = st.file_uploader("Загрузите PDF", type=["pdf"])
use_llm = st.checkbox(
    "Включить OCR-LLM (использовать LLM для коррекции сильно зашумлённых кусков текста)",
    value=False,
)
prompt_text = st.text_area("Prompt", value=BASE_PROMPT, height=300)

# Запуск обработки
if uploaded and st.button("Запустить"):
    st.info("⚠️ Обработка документа может занять некоторое время. Пожалуйста, дождитесь завершения.")

    # Временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getvalue())
        pdf_path = tmp.name

    progress_text = st.empty()
    progress_bar = st.progress(0)

    # Колбэк для отображения прогресса
    def cb(progress: float, desc: str) -> None:
        progress_bar.progress(int(progress * 100))
        progress_text.text(desc)

    # Запуск парсинга
    result: Dict[str, Any] = parse_document(
        pdf_path=pdf_path,
        use_llm=use_llm,
        prompt=prompt_text,
        progress_cb=cb,
    )

    os.unlink(pdf_path)

    # Отображение результатов
    st.subheader("Распознанные поля")
    st.json(result["fields"])

    # Выгрузка в Excel
    df = pd.DataFrame([
        {"field": k, "value": v.get("value"), "location": v.get("location")}
        for k, v in result["fields"].items()
    ])
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    st.download_button(
        "Скачать как Excel",
        data=buf.getvalue(),
        file_name="fields.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # OCR-страницы
    st.subheader("OCR страницы")
    for page in result["pages"]:
        st.image(page["info"]["overlay"], caption=f"Страница {page['page']}")
        with st.expander(f"Блоки страницы {page['page']}"):
            for block in page["info"]["blocks"]:
                st.image(block["crop_data"], caption=block["final"]["text"])
