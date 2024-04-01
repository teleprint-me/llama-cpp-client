"""
Module: llama_cpp_client.history
"""

import json
import os
from pathlib import Path
from typing import Dict, List

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory


class LlamaCppHistory:
    """Track the language models conversational history"""

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

        # Define the list for tracking chat messages.
        # Each message is a dictionary with the following structure:
        # {"role": "user/assistant/system", "content": "<message content>"}
        self._messages: List[Dict[str, str]] = []
        # Set the system message, if any. There is only one system message and
        # it is always the first element within a sequence of messages.
        # self.messages = [{"role": "system", "content": value}]
        self.system_message = system_message

    def __len__(self) -> int:
        return len(self.messages)

    def __contains__(self, message: Dict[str, str]) -> bool:
        return message in self.messages

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self.messages[index]

    def __setitem__(self, index: int, message: Dict[str, str]) -> None:
        self.messages[index] = message

    @property
    def messages(self) -> List[Dict[str, str]]:
        return self._messages

    @property
    def system_message(self) -> Dict[str, str]:
        return self._system_message

    @system_message.setter
    def system_message(self, content: str) -> None:
        if content is None:
            raise ValueError("Content cannot be None for system_message.")

        self._system_message = {"role": "system", "content": content}

        if self._messages:
            self._messages[0] = self._system_message
        else:
            self._messages = [self._system_message]

    def load(self) -> List[Dict[str, str]]:
        """Load the language models previous session"""
        try:
            with open(self.file_path, "r") as chat_session:
                self.messages = json.load(chat_session)
            print(f"LlamaCppHistory: Using cache: {self.file_path}")
            return self.messages
        except (FileNotFoundError, json.JSONDecodeError):
            self.save()  # create the missing file
            print(f"LlamaCppHistory: Created new cache: {self.file_path}")

    def save(self) -> None:
        """Save the language models current session"""
        try:
            with open(self.file_path, "w") as chat_session:
                json.dump(self.messages, chat_session, indent=2)
            print(f"LlamaCppHistory: Saved cache: {self.file_path}")
        except TypeError as e:
            print(f"LlamaCppHistory: Cache failed: {e}")

    def append(self, message: Dict[str, str]) -> None:
        """Append a message into the language models current session"""
        self.messages.append(message)

    def insert(self, index: int, element: object) -> None:
        """Insert a message into the language models current session"""
        self.messages.insert(index, element)

    def pop(self, index: int = None) -> Dict[str, str]:
        """Pop a message from the language models current session"""
        # Guard the models system message
        if self.system_message and index == 0:
            raise IndexError("System message is at index 0 and cannot be popped")
        # Use default pop if index is None
        if index is None:
            return self.messages.pop()
        # Return the popped message
        return self.messages.pop(index)  # Raises index error if index is out of bounds

    def replace(self, index: int, content: str) -> None:
        """Substitute a message within the language models current session"""
        try:
            self.messages[index]["content"] = content
        except (IndexError, KeyError) as e:
            print(f"ModelHistoryReplace: Failed to substitute chat message: {e}")

    def reset(self) -> None:
        """Reset the language models current session. Warning: This is a destructive action."""
        if self.system_message:
            self.messages = [self.system_message]
        else:
            self.messages = []

    def prompt(self) -> str:
        """Prompt the user for input"""
        return self.session.prompt(
            "> ", auto_suggest=self.auto_suggest, multiline=True
        ).strip()
