"""
Module: llama_cpp_client.function.schemas
"""

GET_CURRENT_WEATHER = {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
}

READ_FILE = {
    "name": "read_file",
    "description": "Reads the contents of a file and returns a substring based on optional line numbers.",
    "parameters": {
        "type": "object",
        "properties": {
            "filepath": {
                "type": "string",
                "description": "The required file path to read from.",
            },
            "start_line": {
                "type": "integer",
                "description": "Optional starting line number to extract from. Default is 0.",
            },
            "end_line": {
                "type": "integer",
                "description": "Optional ending line number to extract up to. If not provided or set to null, it defaults to the last line in the file.",
            },
        },
        "required": ["filepath"],
    },
}

BUILTIN_FUNCTION_SCHEMAS = [
    GET_CURRENT_WEATHER,
    READ_FILE,
]
