import argparse
from . import extract_key_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract key fields from a contract image"
    )
    parser.add_argument("--image", required=True, help="Path to document image")
    parser.add_argument(
        "--backend", default="openrouter", help="LLM backend (openrouter/local)"
    )
    args = parser.parse_args()

    data = extract_key_lines(args.image, llm_backend=args.backend)
    print(data)


if __name__ == "__main__":
    main()
