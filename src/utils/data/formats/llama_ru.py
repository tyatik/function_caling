from typing import List, Dict, Any
import json
from torch.nn.modules.pixelshuffle import F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from transformers.models import patchtsmixer
from tqdm import tqdm

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

def paths_to_key(d, key):
  paths = []
  if key in d:
    paths.append([key])
  for k in d.keys():
    if type(d[k]) == type({}):
      local_paths = paths_to_key(d[k], key)
      l_p_ex = [[k] + p for p in local_paths]
      paths.extend(l_p_ex)
  return paths

def children_paths(d):
   paths = []
   if type(d) == type({}):
    for k in d.keys():
        if type(d[k]) == type({}) or type(d[k]) == type([]):
          local_paths = [[k] + p for p in children_paths(d[k])]
          paths.extend(local_paths)
        else:
          paths.append([k])
   elif type(d) == type([]):
    local_path = [[i] + p for i in range(len(d)) for p in children_paths(d[i])]
   return paths

def value_by_path(d, path):
  el = d
  for p in path:
    el = el[p]
  return str(el)

def set_value_by_path(d, path, value):
  el = d
  for p in path[:-1]:
    el = el[p]
  el[path[-1]] = value

def create_batches(m_texts, batch_size):
  m_text_batches = []
  batch = []
  for i in range(len(m_texts)):
    batch.append(m_texts[i])
    if (i + 1) % batch_size == 0 or i == len(m_texts) - 1:
      m_text_batches.append(batch)
      batch = []
  return m_text_batches

def translate_batches(batches, tokenizer, model, device, desc):
  batches_ru = []
  for b in tqdm(batches, desc=desc):
    input_ids = tokenizer(b, return_tensors="pt", padding="max_length", truncation=True)
    generated_tokens = model.generate(**input_ids.to(device))
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    batches_ru.append(result)
  return batches_ru

def convert(messages: List[str], functions: List[str]) -> List[str]:
    m_dicts_base = [json.loads(m) for m in messages]
    m_paths_base = []
    m_texts = []
    fc_paths_base = []
    fc_args = []
    fr_paths_base = []
    fr_args = []
    for i, m in enumerate(m_dicts_base):
       for j, d in enumerate(m):
          if d["role"] == "function_call":
             data = json.loads(d["content"])["arguments"]
             paths = [[i, j, p] for p in children_paths(data)]
             args = [value_by_path(data, p[2]) for p in paths]
             fc_paths_base.extend(paths)
             fc_args.extend(args)
          elif d["role"] == "function_response":
             data = json.loads(d["content"])
             paths = [[i, j, p] for p in children_paths(data)]
             args = [value_by_path(data, p[2]) for p in paths]
             fr_paths_base.extend(paths)
             fr_args.extend(args)
          else:
             m_paths_base.append([i, j])
             m_texts.append(d["content"])
    

    f_dicts_base = [json.loads(f) for f in functions]
    f_paths_base = [[f, d] for f in range(len(f_dicts_base)) for d in range(len(f_dicts_base[f]))]
    f_descriptions = [d["description"] for f in f_dicts_base for d in f]

    p_paths_base = [[i, j, p] for i, f in enumerate(f_dicts_base) for j, d in enumerate(f) for p in paths_to_key(d, "description")]
    p_descriptions = [value_by_path(f_dicts_base[i][j], p) for i, j, p in p_paths_base]

    with open("/content/function_caling/src/utils/data/formats/llama_ru_config.json", "r") as fp:
        config = json.load(fp)

    batch_size = config["batch_size"]

    m_text_batches = create_batches(m_texts, batch_size)
    fc_args_batches = create_batches(fc_args, batch_size)
    fr_args_batches = create_batches(fr_args, batch_size)
    f_description_batches = create_batches(f_descriptions, batch_size)
    p_description_batches = create_batches(p_descriptions, batch_size)
    

    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
    device = config["device"]
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    prefix = 'translate to ru: '
    prefix = ""

    m_text_batches_ru = translate_batches(m_text_batches, tokenizer, model, device, "Translating messages")
    m_text_ru = [el for b in m_text_batches_ru for el in b]
    fc_args_batches_ru = translate_batches(fc_args_batches, tokenizer, model, device, "Translating function calls")
    fc_args_ru = [el for b in fc_args_batches_ru for el in b]
    fr_args_batches_ru = translate_batches(fr_args_batches, tokenizer, model, device, "Translating function responses")
    fr_args_ru = [el for b in fr_args_batches_ru for el in b]
    f_description_batches_ru = translate_batches(f_description_batches, tokenizer, model, device, "Translating function descriptions")
    f_descriptions_ru = [el for b in f_description_batches_ru for el in b]
    p_description_batches_ru = translate_batches(p_description_batches, tokenizer, model, device, "Translating arg descriptions")
    p_descriptions_ru = [el for b in p_description_batches_ru for el in b]

    for i in range(len(m_paths_base)):
      path = m_paths_base[i]
      text = m_text_ru[i]
      m_dicts_base[path[0]][path[1]]["content"] = text

    for i in range(len(f_paths_base)):
      path = f_paths_base[i]
      text = f_descriptions_ru[i]
      f_dicts_base[path[0]][path[1]]["description"] = text

    for i in range(len(p_paths_base)):
      path = p_paths_base[i]
      text = p_descriptions_ru[i]
      f_link = f_dicts_base[path[0]][path[1]]
      set_value_by_path(f_link, path[2], text)

    for i in range(len(fc_paths_base)):
      path = fc_paths_base[i]
      text = fc_args_ru[i]
      f_link = json.loads(m_dicts_base[path[0]][path[1]]["content"])
      if type(f_link) == type({}):
        if "arguments" in f_link:
          set_value_by_path(f_link["arguments"], path[2], text)
      m_dicts_base[path[0]][path[1]]["content"] = json.dumps(f_link)

    for i in range(len(fr_paths_base)):
      path = fr_paths_base[i]
      text = fr_args_ru[i]
      f_link = json.loads(m_dicts_base[path[0]][path[1]]["content"])
      set_value_by_path(f_link, path[2], text)
      m_dicts_base[path[0]][path[1]]["content"] = json.dumps(f_link)
    
    result = {"text":[]}
    for local_messages, local_functions in zip(m_dicts_base, f_dicts_base):
      tools = ",\n".join([json.dumps(function, indent=4) for function in local_functions])
      local_messages[0]["content"] = SYSTEM_PROMPT.format(tools=tools)
      
      messages_string = [S_B, INST_B]
      for message in local_messages:
          if messages_string[-1] == S_E:
              messages_string.append(S_B)
              messages_string.append(INST_B)

          if message["role"] == "system":
              messages_string.append(SYS_B)
              messages_string.append(message["content"])
              messages_string.append(SYS_E)
          elif message["role"] == "user":
              messages_string.append(message["content"])
              messages_string.append(INST_E)
          elif message["role"] == "assistant":
              messages_string.append(message["content"])
              messages_string.append(S_E)
          elif message["role"] == "function_call":
              messages_string.append(TOOL_CALL_B)
              messages_string.append(message["content"])
              messages_string.append(TOOL_CALL_E)
          elif message["role"] == "function_response":
              messages_string.append(TOOL_RESPONSE_B)
              messages_string.append(message["content"])
              messages_string.append(TOOL_RESPONSE_E)
      result["text"].append("".join(messages_string))
    
    return result

