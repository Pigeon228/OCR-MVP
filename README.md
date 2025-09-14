Важно: вставте ключ для демо на 3 доллара в config.py добавив в конце ключа цифру 0: sk-or-v1-91c9c5047dd012be21031b2f7cb924f77988566472973dd930bf200e6c17e0e

# OCR-MVP

Минимальный прототип для распознавания текста и извлечения ключевых полей из PDF документов.

## Компоненты
- **text_recognition** — OCR на базе EasyOCR с опциональной проверкой результата через LLM.
- **document_parser** — постранично обрабатывает PDF и выделяет поля договора с помощью LLM.
- **llm** — роутер для подключения OpenRouter или локальной LLM.
- **streamlit_app.py** — веб‑демо на Streamlit.

## Установка
1. Убедитесь, что установлен Python 3.10+
2. Создайте виртуальное окружение и установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Укажите ключ API OpenRouter в `config.py` или через переменную `OPENROUTER_API_KEY`.

## Использование
### OCR картинки
```bash
python -m text_recognition --image path/to/image.png --output outputs/
```
### Парсинг PDF
```bash
python -m document_parser --pdf path/to/file.pdf
```
### Веб-интерфейс
```bash
streamlit run streamlit_app.py
```

Результаты распознавания сохраняются в указанные каталоги, а разметка страниц отображается в Streamlit-приложении.
