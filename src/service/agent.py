import yaml
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel, PeftConfig
import re

class Environment():
    def __init__(self):
        self.menu = [
            {"name":"Пиво","cost":5},
            {"name":"Рыбка","cost":10},
        ]
        self.tables = [
            {"number":1, "is_free":False},
            {"number":2, "is_free":True},
        ]

        self.functions = [
            """{ "name": "buy_item", "description": "Покупка еды или напитка из меню", "parameters": { "type": "object", "properties": { "name": { "type": "string", "description": "Название позиции из меню, которую нужно приобрести" }}, "required": [ "name" ] } }""",
            """{ "name": "book_table", "description": "Бронирование столика в таверне", "parameters": { "type": "object", "properties": { "number": { "type": "string", "description": "Номер столика, который нужно забронировать" }}, "required": [ "number" ] } }"""
        ]
    def buy_item(self, name: str):
        is_enable = "Нет"
        cost = 0
        for i in self.menu:
            if i["name"] == name:
                is_enable = "Да"
                cost = i["cost"]
                break
        result = {
            "Есть в наличии":is_enable,
            "Стоимость":cost,
        }
        return result
    def book_table(self, number: str):
        number = int(number)
        answer = "Нет такого столика"
        for t in self.tables:
            if t["number"] == number:
                if t["is_free"] == True:
                    answer = "Успешно"
                else:
                    answer = "Столик занят"
        result = {
            "Ответ":answer
        }
        return result
    def default_func(self):
        return "Не удалось получить ответ."
    
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords:list, tokenizer):
        self.keywords = keywords
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        result = False
        for key in self.keywords:
          text = self.tokenizer.batch_decode(input_ids)[0]
          if text.endswith(key):
            result = True
        return result

class Agent():
    def __init__(self, env, config):
        self.env = env
        
        peft_config = PeftConfig.from_pretrained(config["model_args"]["model_name"])
        model_config = {
            "pretrained_model_name_or_path":peft_config.base_model_name_or_path,
            "low_cpu_mem_usage":True,
            "torch_dtype":torch.float16,
            "device_map":"auto",
            "offload_state_dict":True,
            "quantization_config":BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        }
        base_model = AutoModelForCausalLM.from_pretrained(**model_config)
        self.model = PeftModel.from_pretrained(base_model, config["model_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        self.generation_config = GenerationConfig.from_pretrained(model_config["pretrained_model_name_or_path"])
        self.generation_config.max_new_tokens = 1000
        self.stopping_criteria = KeywordsStoppingCriteria(["</TOOL_CALL>", "</s>"], self.tokenizer)

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

        self.prompt = f"{S_B}{SYS_B}{SYSTEM_PROMPT.format(tools=json.dumps(env.functions, ensure_ascii=False))}{SYS_E}"
        self.history = self.prompt

    def interract(self, message: str):
        inputs = self.history + message
        outputs = self.generate(inputs)
        self.history = self.history + outputs

        function_call = re.findall("<TOOL_CALL>(.*)</TOOL_CALL>", outputs)
        if len(function_call) > 0:
            func_name, func_args = self.parse_function(function_call[0])
        response = getattr(self.env, func_name, self.env.default_func)(**func_args)
        response = f"<TOOL_RESPONSE>{response}</TOOL_RESPONSE>"

        self.history = self.history + response
        answer = self.generate(self.history)
        self.history = self.history + answer
        return answer

    def generate(self, prompt):
        data = self.tokenizer(prompt, return_tensors="pt")
        data = {k: v.to(self.model.device) for k, v in data.items()}
        output_ids = self.model.generate(
            **data,
            generation_config=self.generation_config,
            stopping_criteria=StoppingCriteriaList([self.stopping_criteria])
        )[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output
    
    def parse_function(self, function_call_string):
        try:
            fc = json.loads(function_call_string)
            name = fc["name"]
            args = fc["arguments"]
            return name, args
        except:
            return "default_func", {}

config_path = "configs/inference.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

ENVIRONMENT = Environment()
AGENT = Agent(ENVIRONMENT, config)
