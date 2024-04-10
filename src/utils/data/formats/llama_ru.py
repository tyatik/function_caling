from typing import List, Dict, Any
import json
import re

SYSTEM_PROMPT = (
    "Ты - незаменимый помощник, выполняющий задачу вызова функции. Тебе предоставлены сигнатуры функций, заключенные в xml теги <TOOLS></TOOLS>"
    "Ты можешь вызывать одну или несколько функций по запросу пользователя. Не придумывай значения аргументов, если они не указаны пользователем."
    "Вызывай одну из следующих функций:\n"
    "<TOOLS>\n"
    "{tools}\n"
    "</TOOLS>\n\n"
    "Для каждого вызова функции возвращай названия функций и аргументы в формате JSON. Результат запиши внутри тегов <TOOL_CALL></TOOL_CALL> вот так: "
    "<TOOL_CALL>\n"
    "{{\"name\": <function-name>, \"arguments\": <args-dict>}}\n"
    "</TOOL_CALL>\n"
    "После вызова функции ты получишь результат вызова внутри тегов <TOOL_RESPONSE></TOOL_RESPONSE>."
    "Ответь на запрос пользователя на основе результата вызова функции."
)

S_B, S_E = "<s>", "</s>"
INST_B, INST_E = "[INST] ", " [/INST] "
SYS_B, SYS_E = "<<SYS>>\n", "\n<</SYS>>\n\n"
TOOL_CALL_B, TOOL_CALL_E = "<TOOL_CALL>\n", "\n</TOOL_CALL>\n\n"
TOOL_RESPONSE_B, TOOL_RESPONSE_E = "<TOOL_RESPONSE>\n", "\n</TOOL_RESPONSE>\n\n"


def row_to_tokens(row) -> List[str]:
    messages = json.loads(row["conversations_ru"])
    content = {
        "messages":[],
        "functions":[]
    }
    for m in messages:
        if m["role"] == "system":
            c = m["content"].lsplit("{", 1)
            f = re.findall(r"\{\}", m["content"])
            content["functions"] = f
            new_m = {
                "role":"system",
                "content":c
            }
            content["messages"].append(new_m)
        elif m["role"] == "user":
            content["messages"].append(m)
        elif m["role"] == "assistant":
            if m["content"] == None:
                new_m = {
                    "role":"function_call",
                    "content":json.dumps(m["function_call"], ensure_ascii=False)
                }
                content["messages"].append(new_m)
            else:
                content["messages"].append(m)
        elif m['role'] == "function":
            c = {
                "name":m["name"],
                "arguments":m["content"]
            }
            new_m = {
                "role":"function_response",
                "content":json.dumps(c, ensure_ascii=False)
            }
            content["messages"].append(new_m)

    messages = json.loads(content["messages"])
    functions = json.loads(content["functions"])

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

