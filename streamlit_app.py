import io
import os
import tempfile
from typing import Dict, Any

import pandas as pd
import streamlit as st

from document_parser import parse_document, BASE_PROMPT

st.set_page_config(page_title="OCR Demo")
st.title("OCR-MVP Demo")

uploaded = st.file_uploader("Загрузите PDF", type=["pdf"])
use_llm = st.checkbox("Включить OCR-LLM", value=False)
prompt_text = st.text_area("Prompt", value=BASE_PROMPT, height=300)

if uploaded and st.button("Запустить"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getvalue())
        pdf_path = tmp.name
    progress_text = st.empty()
    progress_bar = st.progress(0)

    def cb(progress: float, desc: str) -> None:
        progress_bar.progress(int(progress * 100))
        progress_text.text(desc)

    result: Dict[str, Any] = parse_document(
        pdf_path=pdf_path,
        use_llm=use_llm,
        prompt=prompt_text,
        progress_cb=cb,
    )
    os.unlink(pdf_path)

    st.subheader("Распознанные поля")
    st.json(result["fields"])

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

    st.subheader("OCR страницы")
    for page in result["pages"]:
        st.image(page["info"]["overlay"], caption=f"Страница {page['page']}")
        with st.expander(f"Блоки страницы {page['page']}"):
            for block in page["info"]["blocks"]:
                st.image(block["crop_data"], caption=block["final"]["text"])
