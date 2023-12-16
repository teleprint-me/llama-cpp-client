"""
examples/extract_chat_template.py: 
    Example file to experiment with extracting the chat template from the models metadata.
"""
from __future__ import annotations

from gguf import GGUFReader, Keys


def get_chat_template(model_file: str) -> str:
    reader = GGUFReader(model_file)

    # Access the 'chat_template' field directly using its key
    chat_template_field = reader.fields[Keys.Tokenizer.CHAT_TEMPLATE]

    # Extract the chat template string from the field
    chat_template_memmap = chat_template_field.parts[-1]
    chat_template_string = chat_template_memmap.tobytes().decode("utf-8")

    return chat_template_string


def main() -> None:
    # this is just an exercise to determine how it might be done in practice
    model_file = "models/mistralai/Mixtral-8x7B-Instruct-v0.1/Mixtral-8x7B-Instruct-v0.1-q4_0.gguf"
    chat_template = get_chat_template(model_file)
    print(chat_template)


if __name__ == "__main__":
    main()
