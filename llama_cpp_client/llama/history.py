"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client.llama.history

Description: Module for handling language models completion history.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory


class LlamaCppHistory:
    """Track the language models completions history"""

    def __init__(self, session_name: str, system_message: str = None):
        # Define the cache path for storing chat history
        home = os.environ.get("HOME", ".")  # get user's home path, else assume cwd
        cache = Path(f"{home}/.cache/llama-cpp-client")  # set the cache path
        cache.mkdir(parents=True, exist_ok=True)  # ensure the directory exists

        # Define the file path for storing chat history
        self.file_path = cache / f"{session_name}.json"

        # Define the file path for storing prompt session history
        file_history_path = cache / f"{session_name}.history"
        self.session = PromptSession(history=FileHistory(file_history_path))
        self.auto_suggest = AutoSuggestFromHistory()

        # Define the list for tracking chat completions.
        # Each element is a dictionary with the following structure:
        # completions:
        #   {"role": "user/assistant/system", "content": "prompt/completion"}
        self._completions: List[Dict[str, str]] = []

        # Set the system message, if any. There is only one system message and
        # it is always the first element within a sequence of completions.
        # self._completions = [{"role": "system", "content": value}]
        if system_message is not None:
            system_message_path = Path(system_message)
            if system_message_path.exists():
                system_message = system_message_path.open("r").read().rstrip()
            else:
                system_message = system_message
        self.system_message = system_message

    def __len__(self) -> int:
        return len(self._completions)

    def __contains__(self, message: Dict[str, str]) -> bool:
        return message in self._completions

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self._completions[index]

    def __setitem__(self, index: int, message: Dict[str, str]) -> None:
        self._completions[index] = message

    @property
    def completions(self) -> List[Dict[str, str]]:
        return self._completions

    @property
    def system_message(self) -> Dict[str, str]:
        return self._system_message

    @system_message.setter
    def system_message(self, content: None | str) -> None:
        if content is None:
            # NOTE: Completions don't have system roles
            self._system_message = None
            return  # NOTE: Guard against setting system message

        self._system_message = {"role": "system", "content": content}

        if self._completions:
            self._completions[0] = self._system_message
        else:
            self._completions = [self._system_message]

    def load(self) -> List[Dict[str, str]]:
        """Load the language models previous session"""
        try:
            with open(self.file_path, "r") as chat_session:
                self._completions = json.load(chat_session)
            # print(f"LlamaCppHistory: Using cache: {self.file_path}")
            return self._completions
        except (FileNotFoundError, json.JSONDecodeError):
            self.save()  # create the missing file
            print(f"LlamaCppHistory: Created new cache: {self.file_path}")

    def save(self) -> None:
        """Save the language models current session"""
        try:
            with open(self.file_path, "w") as chat_session:
                json.dump(self._completions, chat_session, indent=2)
            # print(f"LlamaCppHistory: Saved cache: {self.file_path}")
        except TypeError as e:
            print(f"LlamaCppHistory: Cache failed: {e}")

    def append(self, message: Dict[str, str]) -> None:
        """Append a message into the language models current session"""
        self._completions.append(message)

    def insert(self, index: int, element: object) -> None:
        """Insert a message into the language models current session"""
        self._completions.insert(index, element)

    def pop(self, index: int = None) -> Dict[str, str]:
        """Pop a message from the language models current session"""
        # Guard the models system message
        if self.system_message and index == 0:
            raise IndexError("System message is at index 0 and cannot be popped")
        try:
            # Use default pop if index is None
            if index is None:
                return self._completions.pop()
            # Return the popped message
            # NOTE: list.pop raises IndexError if index is out of bounds
            return self._completions.pop(index)
        except IndexError:
            return {}  # No elements exist

    def replace(self, index: int, content: str) -> None:
        """Substitute a message within the language models current session"""
        try:
            self._completions[index]["content"] = content
        except (IndexError, KeyError) as e:
            print(f"ModelHistoryReplace: Failed to substitute chat message: {e}")

    def reset(self) -> None:
        """Reset the language models current session. Warning: This is a destructive action."""
        if self.system_message:
            self._completions = [self.system_message]
        else:
            self._completions = []

    def prompt(self) -> str:
        """Prompt the user for input"""
        return self.session.prompt(
            "> ", auto_suggest=self.auto_suggest, multiline=True
        ).strip()
