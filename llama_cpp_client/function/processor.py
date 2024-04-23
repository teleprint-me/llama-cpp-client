import json
from typing import Any, Callable


def get_function(
    completion: dict[str, str], functions: dict[str, Callable]
) -> dict[str, Any]:
    # Note: the JSON response may not always be valid; be sure to handle errors
    name = completion["function_call"]["name"]
    callback = functions[name]
    arguments = json.loads(completion["function_call"]["arguments"])
    content = callback(**arguments)
    response = {
        "role": "function",
        "name": name,
        "content": content,
    }
    return response
