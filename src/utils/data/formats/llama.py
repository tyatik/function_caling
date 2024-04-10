from typing import List, Dict, Any
import json

SYSTEM_PROMPT = (
    "You are a helpful assistant with function-calling supported. You are provided with function signatures within <TOOLS></TOOLS> XML tags. Use the if required.\n"
    "<TOOLS>\n"
    "{tools}\n"
    "</TOOLS>"
)

S_B, S_E = "<s>", "</s>"
INST_B, INST_E = "[INST] ", " [/INST] "
SYS_B, SYS_E = "<<SYS>>\n", "\n<</SYS>>\n\n"
TOOL_CALL_B, TOOL_CALL_E = "<TOOL_CALL>\n", "\n</TOOL_CALL>\n\n"
TOOL_RESPONSE_B, TOOL_RESPONSE_E = "<TOOL_RESPONSE>\n", "\n</TOOL_RESPONSE>\n\n"


def row_to_tokens(row: Dict[str, Any]) -> List[str]:
    messages = json.loads(row["messages"])
    functions = json.loads(row["functions"])

    # Customize system prompt
    tools = ",\n".join([json.dumps(function, indent=4) for function in functions])
    messages[0]["content"] = SYSTEM_PROMPT.format(tools=tools)

    tokens = [INST_B, SYS_B, messages[0]["content"], SYS_E]
    for i in range(1, len(messages)):
        if i > 1 and messages[i]["role"] == "user":
            tokens.extend([S_E, S_B, INST_B])

        message = messages[i]
        if message["role"] == "user":
            tokens.append(message["content"])
            tokens.append(INST_E)
        elif message["role"] == "assistant":
            tokens.append(message["content"])
        elif message["role"] == "function_call":
            tokens.append(TOOL_CALL_B)
            tokens.append(message["content"])
            tokens.append(TOOL_CALL_E)
        elif message["role"] == "function_response":
            tokens.append(TOOL_RESPONSE_B)
            tokens.append(message["content"])
            tokens.append(TOOL_RESPONSE_E)
    tokens.append(S_E)

    return tokens


def create_llama_test_prompt(row) -> Dict[str, List[str]]:
    tokens = row_to_tokens(row)

    result = {
        "text": [],
        "text_target": [],
    }

    l, r = 0, 0
    for i in range(len(tokens)):
        if tokens[i] == INST_E:
            l = i
        elif tokens[i] == S_E:
            r = i

            # if l == 0:
            #     continue

            result["text"].append("".join(tokens[:l+1]))
            result["text_target"].append("".join(tokens[l+1:r]))

    return result


def create_llama_prompt(row: Dict[str, str]) -> str:
    return "".join(row_to_tokens(row))
