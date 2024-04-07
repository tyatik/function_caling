from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
import json

class FunctionCallingDataset(Dataset):
    def __init__(self, params):
        self.max_input_len = params["max_input_len"]
        self.max_output_len = params["max_output_len"]
        self.tokenizer = params["tokenizer"]
        self.train = params.get("is_train", True)
        self.size = params.get("size",1)

        self.SYSTEM_PROMPT = (
            "You are a helpful assistant with function-calling supported. You are provided with function signatures within <TOOLS></TOOLS> XML tags. "
            "You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. "
            "Here are the available tools:\n"
            "<TOOLS>\n"
            "{tools}\n"
            "</TOOLS>\n\n"
            "For each function call, return a JSON object with the function name and arguments within <TOOL_CALL></TOOL_CALL> XML tags as follows:\n"
            "<TOOL_CALL>\n"
            "{{\"name\": <function-name>, \"arguments\": <args-dict>}}\n"
            "</TOOL_CALL>"
            "You will get function call result within <TOOL_RESPONSE></TOOL_RESPONSE> XML tags. "
            "Answer user query based on the result."
        )

        self.S_B, self.S_E = "<s>", "</s>"
        self.INST_B, self.INST_E = "[INST] ", " [/INST] "
        self.SYS_B, self.SYS_E = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.TOOL_CALL_B, self.TOOL_CALL_E = "<TOOL_CALL>\n", "\n</TOOL_CALL>\n\n"
        self.TOOL_RESPONSE_B, self.TOOL_RESPONSE_E = "<TOOL_RESPONSE>\n", "\n</TOOL_RESPONSE>\n\n"

        hf_dataset = load_dataset("korotkov/glaive-function-calling-v2-parsed")

        # Convert dataset to another format
        dataset_key = "train" if self.train else "test"
        self.dataset = []
        self.prompts = []
        self.answers = []
        for row in tqdm(hf_dataset[dataset_key]):
            messages = json.loads(row["messages"])
            functions = json.loads(row["functions"])
            all_text, prompt, answer = self.llama_convert(messages, functions)
            self.dataset.append(all_text)
            self.prompts.append(prompt)
            self.answers.append(answer)
        self.dataset = self.dataset[:int(len(self.dataset)*self.size)]
        self.prompts = self.prompts[:int(len(self.prompts)*self.size)]
        self.answers = self.answers[:int(len(self.answers)*self.size)]

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, ind):
        text = self.dataset[ind]
        val_text = self.prompts[ind]
        out_text = self.answers[ind]
        item = {
            "train":{
                "input_text":text,
                "output_text":text
            },
            "val":{
                "input_text":val_text
            },
            "labels":{
                "output_text":out_text
            }
        }
        return item
    def show_samples(self, indexes):
        texts = [self.prompts[ind] for ind in indexes]
        print("Prompts:")
        for t in texts:
          print(t)
        return texts
    def llama_convert(self, messages, functions):
      tools = ",\n".join([json.dumps(function, indent=4) for function in functions])
      messages[0]["content"] = self.SYSTEM_PROMPT.format(tools=tools)

      messages_string = [self.S_B, self.INST_B]
      prompts_string = [self.S_B, self.INST_B]
      answers_string = []
      for message in messages:
          if message["role"] == "system":
              messages_string.append(self.SYS_B)
              messages_string.append(message["content"])
              messages_string.append(self.SYS_E)

              prompts_string.append(self.SYS_B)
              prompts_string.append(message["content"])
              prompts_string.append(self.SYS_E)
          elif message["role"] == "user":
              messages_string.append(message["content"])
              messages_string.append(self.INST_E)

              prompts_string.append(message["content"])
              prompts_string.append(self.INST_E)
          elif message["role"] == "assistant":
              messages_string.append(message["content"])
              messages_string.append(self.S_E)

              answers_string.append(message["content"])
              answers_string.append(self.S_E)
          elif message["role"] == "function_call":
              messages_string.append(self.TOOL_CALL_B)
              messages_string.append(message["content"])
              messages_string.append(self.TOOL_CALL_E)

              answers_string.append(self.TOOL_CALL_B)
              answers_string.append(message["content"])
              answers_string.append(self.TOOL_CALL_E)
          elif message["role"] == "function_response":
              messages_string.append(self.TOOL_RESPONSE_B)
              messages_string.append(message["content"])
              messages_string.append(self.TOOL_RESPONSE_E)

              answers_string.append(self.TOOL_RESPONSE_B)
              answers_string.append(message["content"])
              answers_string.append(self.TOOL_RESPONSE_E)

      all_text = "".join(messages_string)
      spl = all_text.rsplit(self.INST_E, 1)
      prompt = spl[0] + self.INST_E
      answer = spl[1]
      return all_text, prompt, answer
    def collate_fn(self, batch):
        texts = ["input_text", "output_text"]
        lens = [self.max_input_len, self.max_output_len]

        new_batch = {
            "train":{
                "input_text":None,
                "output_text":None
            },
            "val":{
                "input_text":None
            },
            "labels":{
                "output_text":None
            }
        }

        for t, l in zip(texts, lens):
            max_batch_len = 0
            texts = [b["train"][t] for b in batch]
            for text in texts:
                tokenized = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
                length = tokenized["input_ids"].shape[1]
                if length > max_batch_len:
                    max_batch_len = length
            max_batch_len = min(max_batch_len, l)
            for k in new_batch.keys():
                tokenized = self.tokenizer(texts, return_tensors="pt", max_length=max_batch_len, padding="max_length", truncation=True)
                if t == "input_text":
                  tokenized["input_ids"] = tokenized["input_ids"][:,:-1]
                  tokenized["attention_mask"] = tokenized["attention_mask"][:,:-1]
                else:
                  tokenized["input_ids"] = tokenized["input_ids"][:,1:]
                  tokenized["attention_mask"] = tokenized["attention_mask"][:,1:]
                if t in new_batch[k]:
                    if k != "train":
                      true_texts = [b[k][t] for b in batch]
                      tokenized = self.tokenizer(true_texts, padding=True, return_tensors="pt")
                    new_batch[k][t] = tokenized

        return new_batch