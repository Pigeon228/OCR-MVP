import argparse

from . import process_image


def main():
    parser = argparse.ArgumentParser(description="Run OCR on an image")
    parser.add_argument(
        "--image",
        default="examples/inputs/contract3.png",
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        default="examples/outputs",
        help="Directory to store results",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM verification",
    )
    parser.add_argument(
        "--backend",
        default="openrouter",
        help="LLM backend name",
    )
    args = parser.parse_args()

    info = process_image(
        image_path=args.image,
        output_dir=args.output,
        use_llm=args.llm,
        llm_backend=args.backend,
    )

    print(f"✅ Принято {info['kept']} блоков")
    print(f"🖼 Оверлей: {info['overlay_path']}")
    print(f"📄 Easy (сырое): {info['easy_txt']}")
    print(f"📄 Итог (после LLM): {info['verified_txt']}")
    print(f"🧾 Лог блоков: {info['blocks_json']}")
    print(f"🖼 Кропы: {info['crops_dir']}")


if __name__ == "__main__":
    main()
