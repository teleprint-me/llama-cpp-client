"""
llama_cpp_client/format.py
"""
from typing import Any, Callable, Dict, List, Optional, Protocol

import jinja2
from gguf import GGUFReader, Keys
from jinja2 import Template

from llama_cpp_client.singleton import Singleton
from llama_cpp_client.types import ChatMessage

# NOTE: We sacrifice readability for usability.
# It will fail to work as expected if we attempt to format it in a readable way.
llama2_template = """{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]\n{% elif message['role'] == 'assistant' %}{{ message['content'] }}\n{% elif message['role'] == 'system' %}<<SYS>> {{ message['content'] }} <</SYS>>\n{% endif %}{% endfor %}"""


def get_prompt_sequence(messages: List[ChatMessage]) -> str:
    """
    Converts a sequence of chat messages into a single formatted string, omitting the role.

    :param messages: A list of chat messages with roles and content.
    :return: A formatted string representing the conversation content.
    """
    return "".join([msg["content"] for msg in messages])


# Base Chat Formatter Protocol
class ChatFormatterInterface(Protocol):
    def __init__(self, template: Optional[object] = None):
        ...

    def __call__(
        self,
        messages: List[ChatMessage],
        **kwargs,
    ) -> str:
        ...

    @property
    def template(self) -> str:
        ...

    def get_chat_template(model_file: str) -> str:
        ...


class AutoChatFormatter(ChatFormatterInterface):
    def __init__(
        self,
        model_file: Optional[str] = None,
        template: Optional[str] = None,
        template_class: Optional[Template] = None,
    ):
        if model_file is not None:
            self._reader = GGUFReader(model_file)

        if template is not None:
            self._template = template
        else:
            self._template = self.get_chat_template()

        self._environment = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
        ).from_string(
            self._template,
            template_class=template_class,
        )

    def __call__(
        self,
        messages: List[ChatMessage],
        **kwargs: Any,
    ) -> str:
        return self._environment.render(messages=messages, **kwargs)

    @property
    def template(self) -> str:
        return self._template

    def get_chat_template(self) -> str:
        # Access the 'chat_template' field directly using its key
        chat_template_field = self._reader.fields[Keys.Tokenizer.CHAT_TEMPLATE]

        # Extract the chat template string from the field
        chat_template_memmap = chat_template_field.parts[-1]
        chat_template_string = chat_template_memmap.tobytes().decode("utf-8")

        return chat_template_string


class FormatterNotFoundException(Exception):
    pass


class ChatFormatterFactory(Singleton):
    _chat_formatters: Dict[str, Callable[[], ChatFormatterInterface]] = {}

    def register_formatter(
        self,
        name: str,
        formatter_callable: Callable[[], ChatFormatterInterface],
        overwrite=False,
    ):
        if not overwrite and name in self._chat_formatters:
            raise ValueError(
                f"Formatter with name '{name}' is already registered. Use `overwrite=True` to overwrite it."
            )
        self._chat_formatters[name] = formatter_callable

    def unregister_formatter(self, name: str):
        if name in self._chat_formatters:
            del self._chat_formatters[name]
        else:
            raise ValueError(f"No formatter registered under the name '{name}'.")

    def get_formatter_by_name(self, name: str) -> ChatFormatterInterface:
        try:
            formatter_callable = self._chat_formatters[name]
            return formatter_callable()
        except KeyError:
            raise FormatterNotFoundException(
                f"Invalid chat format: {name} (valid formats: {list(self._chat_formatters.keys())})"
            )


# Define a chat format class
class Llama2Formatter(AutoChatFormatter):
    def __init__(self):
        super().__init__(llama2_template)


# Mistral and Llama have identical chat formats
class MistralFormatter(AutoChatFormatter):
    def __init__(self):
        super().__init__(llama2_template)


# With the Singleton pattern applied, regardless of where or how many times
# ChatFormatterFactory() is called, it will always return the same instance
# of the factory, ensuring that the factory's state is consistent throughout
# the application.
ChatFormatterFactory().register_formatter("llama2", Llama2Formatter)
ChatFormatterFactory().register_formatter("mistral", MistralFormatter)
