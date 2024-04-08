from typing import List, Dict, Any
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

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


def convert(messages: List[Dict[str, Any]], functions: List[str]) -> str:
    tools = ",\n".join([json.dumps(function, indent=4) for function in functions])
    messages[0]["content"] = SYSTEM_PROMPT.format(tools=tools)
    with open("src/utils/data/formats/llama_ru_config.json", "r") as fp:
        config = json.load(fp)

    model = T5ForConditionalGeneration.from_pretrained(config["model_name"])
    device = config["device"]
    model.to(device)

    tokenizer = T5Tokenizer.from_pretrained(config["model_name"])
    prefix = 'translate to ru: '

    messages_string = [S_B, INST_B]
    for message in messages:
        if messages_string[-1] == S_E:
            messages_string.append(S_B)
            messages_string.append(INST_B)

        if message["role"] == "system":
            messages_string.append(SYS_B)
            
            text = prefix + message["content"]
            input_ids = tokenizer(text, return_tensors="pt")
            generated_tokens = model.generate(**input_ids.to(device))
            result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            messages_string.append(result)

            messages_string.append(SYS_E)
        elif message["role"] == "user":
            
            text = prefix + message["content"]
            input_ids = tokenizer(text, return_tensors="pt")
            generated_tokens = model.generate(**input_ids.to(device))
            result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            messages_string.append(result)
            
            messages_string.append(INST_E)
        elif message["role"] == "assistant":
            
            text = prefix + message["content"]
            input_ids = tokenizer(text, return_tensors="pt")
            generated_tokens = model.generate(**input_ids.to(device))
            result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            messages_string.append(result)
            
            messages_string.append(S_E)
        elif message["role"] == "function_call":
            messages_string.append(TOOL_CALL_B)
            messages_string.append(message["content"])
            messages_string.append(TOOL_CALL_E)
        elif message["role"] == "function_response":
            messages_string.append(TOOL_RESPONSE_B)
            messages_string.append(message["content"])
            messages_string.append(TOOL_RESPONSE_E)
    
    return {"text": "".join(messages_string)}