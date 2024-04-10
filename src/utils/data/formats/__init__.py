import json

from .llama import (
    create_llama_prompt,
    create_llama_test_prompt,
    TOOL_CALL_B as LLAMA_TOOL_CALL_B,
    TOOL_CALL_E as LLAMA_TOOL_CALL_E,
)

FORMATS_DICT = {
    "raw": {
        "train": lambda row: json.dumps(row),
        "test": lambda row: json.dumps(row),
        "tool_call_b": "<TOOL_CALL>",
        "tool_call_e": "</TOOL_CALL>",
    },
    "llama": {
        "train": create_llama_prompt,
        "test": create_llama_test_prompt,
        "tool_call_b": LLAMA_TOOL_CALL_B,
        "tool_call_e": LLAMA_TOOL_CALL_E,
    },
}
