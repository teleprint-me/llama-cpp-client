"""
Module: llama_cpp_client.types
"""
from typing import List, Literal, TypedDict


# Define a custom ChatMessage data structure
class ChatMessage(TypedDict):
    """
    Represents a message in a chat conversation.

    Attributes:
        - role: The role of the message sender (e.g., "system", "assistant", "function", "user").
        - content: The content of the message.
    """

    role: Literal["system", "assistant", "function", "user"]
    content: str


# Define a custom ChatChoice data structure
class ChatChoice(TypedDict):
    """
    Represents a choice within a chat completion.

    Attributes:
        - index: The index of the choice.
        - message: The message content of the choice.
        - finish_reason: The reason for finishing (optional).
    """

    index: int
    message: ChatMessage
    finish_reason: str


# Define a custom ChatUsage data structure
class ChatUsage(TypedDict):
    """
    Represents usage information of a chat completion.

    Attributes:
        - prompt_tokens: The number of tokens in the prompt.
        - completion_tokens: The number of tokens in the completion.
        - total_tokens: The total number of tokens used.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# Define a custom Delta data structure
class Delta(TypedDict):
    """
    Represents a delta content which may contain either actual content or a function call.

    Attributes:
        - content: The text content of the delta.
        - function_call: Information about a function call (optional).
        - multimodal: Indicates if the content is multimodal (optional).
        - slot_id: The slot ID (optional).
        - stop: Indicates if the delta should stop (optional).
        - finish_reason: The reason for finishing (optional).
    """

    content: str
    function_call: str
    multimodal: bool
    slot_id: int
    stop: bool
    finish_reason: str


# Define a custom FunctionCall data structure
class FunctionCall(TypedDict):
    """
    Represents a function call which may contain an optional JSON Schema representing arguments.

    Attributes:
        - name: The name of the function to be called (required).
        - arguments: Optional JSON Schema representing arguments to be passed to the function call.
    """

    name: str
    arguments: str


# Define a custom ChatCompletion data structure
class ChatCompletion(TypedDict):
    """
    Represents a chat completion.

    Attributes:
        - id: Unique identifier for the completion.
        - object: Indicates the type of object.
        - created: Timestamp when the completion was created.
        - model: The model used for the completion.
        - system_fingerprint: A system fingerprint identifier.
        - choices: List of completion choices.
        - usage: Usage information.
    """

    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    system_fingerprint: str
    choices: List[ChatChoice]
    usage: ChatUsage


# Define a custom ChatCompletionChunk data structure
class ChatCompletionChunk(TypedDict):
    """
    Represents a chunk of chat completion.

    Attributes:
        - id: Unique identifier for the chunk.
        - object: Indicates the type of object.
        - created: Timestamp when the chunk was created.
        - model: The model used for the completion.
        - system_fingerprint: A system fingerprint identifier.
        - choices: List of completion choices.
    """

    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    system_fingerprint: str
    choices: List[ChatChoice]
