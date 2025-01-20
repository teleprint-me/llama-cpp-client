"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client.llama.client

Description: High-level client for performing language model inference.
"""

from rich import pretty, print
from rich.box import MINIMAL, Box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown, Panel

from llama_cpp_client.llama.api import LlamaCppAPI
from llama_cpp_client.llama.history import LlamaCppHistory


class LlamaCppClient:
    def __init__(
        self,
        api: LlamaCppAPI = None,
        history: LlamaCppHistory = None,
        box: Box = MINIMAL,
    ) -> None:
        # install automatic pretty printing in python repl
        pretty.install()

        # manage llama.cpp client instances
        self.api = api if api is not None else LlamaCppAPI()

        if history is not None:
            self.history = history
        else:
            self.history = LlamaCppHistory(
                session_name="client",
                system_message="My name is Llama. I am a helpful assistant.",
            )

        self.box = box
        self.console = Console()

    def _render_completions_once_on_start(self) -> None:
        self.history.load()
        element = ""
        for completion in self.history:
            if completion["role"] == "user":
                element = ""
                element += completion["content"]
            if completion["role"] == "assistant":
                element += completion["content"]
            markdown = Markdown(element)
            panel = Panel(
                markdown,
                box=self.box,
                title=completion["role"],
                title_align="left",
                highlight=True,
            )
            self.console.print(panel, end="")

    def _render_chat_completions_once_on_start(self) -> None:
        self.history.load()
        for completion in self.history:
            markdown = Markdown(completion["content"])
            panel = Panel(
                markdown,
                box=self.box,
                title=completion["role"],
                title_align="left",
                highlight=True,
            )
            self.console.print(panel)

    def get_token_count(self) -> int:
        token_count = 0
        for completion in self.history.completions:
            if "content" in completion:
                content = completion["content"]
                token_count += len(self.api.tokenize(content))
        return token_count

    def stream_completion(self) -> str:
        # NOTE: The API only supports individual completions at the moment
        # Currently researching how to implement multi-prompting
        content = self.history[-1]["content"]
        generator = self.api.completion(content)
        with Live(
            console=self.console,
            refresh_per_second=30,
            vertical_overflow="visible",
        ) as live:
            for response in generator:
                if "content" in response:
                    token = response["content"]
                    content += token
                    markdown = Markdown(content)
                    panel = Panel(
                        markdown,
                        box=self.box,
                        title="Completion",
                        title_align="left",
                        highlight=True,
                    )
                    live.update(panel, refresh=True)
        return content

    def stream_chat_completion(self) -> str:
        content = ""
        generator = self.api.chat_completion(self.history.completions)
        with Live(
            console=self.console,
            refresh_per_second=30,
            vertical_overflow="visible",
        ) as live:
            for response in generator:
                if "content" in response["choices"][0]["delta"]:
                    token = response["choices"][0]["delta"]["content"]
                    content += token
                    markdown = Markdown(content)
                    panel = Panel(
                        markdown,
                        box=self.box,
                        title="ChatCompletion",
                        title_align="left",
                        highlight=True,
                    )
                    live.update(panel, refresh=True)
        return content

    def prompt_user(self) -> None:
        try:
            self.console.print(Markdown("**user**"))
            prompt = self.history.prompt()
            self.history.append({"role": "user", "content": prompt})
            print()
        # NOTE: Ctrl + C (interrupt) to exit
        except KeyboardInterrupt:
            self.history.save()
            exit()

        # Ctrl + D (eof) to pop a message from history and regenerate completion
        except EOFError:
            if self.history.completions:
                completion = self.history.pop()
                print("\nPopped", completion["role"], "element from history.\n")

    def prompt_assistant(self, completions_type: str) -> None:
        try:
            if completions_type == "completions":
                completion = self.stream_completion()
            elif completions_type == "chat_completions":
                completion = self.stream_chat_completion()
            else:
                raise ValueError(f"Unrecognized completions type: {completions_type}")

            self.history.append({"role": "assistant", "content": completion})
            self.history.save()
            token_count = self.get_token_count()
            print(f"Consuming {token_count} tokens.\n")
        except KeyboardInterrupt:
            if self.history.completions:
                completion = self.history.pop()
                print("\nPopped", completion["role"], "element from history.\n")

    def run(self, completions_type: str):
        if completions_type == "completions":
            self._render_completions_once_on_start()
        elif completions_type == "chat_completions":
            self._render_chat_completions_once_on_start()
        else:
            raise ValueError(f"Unrecognized completions type: {completions_type}")

        while True:
            self.prompt_user()
            self.prompt_assistant(completions_type)

    def run_completions(self) -> None:
        """Convenience method for running completions"""
        self.run("completions")

    def run_chat_completions(self) -> None:
        """Convenience method for running chat completions"""
        self.run("chat_completions")
