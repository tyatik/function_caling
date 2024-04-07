import torch
from torch import nn 
import peft
from peft import PeftModel, PeftConfig, LoraConfig
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

class PeftLLM(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.model_name = params["model_name"]
        self.weights_dtype = params.get("weights_dtype", torch.float16)
        load_in_4bit = params.get("load_in_4bit", False)
        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto',
        low_cpu_mem_usage=True, offload_state_dict=True, torch_dtype=self.weights_dtype, load_in_4bit=load_in_4bit)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.gradient_checkpointing_enable()
        self.base_model.enable_input_require_grads()
        self.tokenizer = params.get("tokenizer", AutoTokenizer.from_pretrained(self.model_name))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.peft_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj"],
            task_type=peft.TaskType.CAUSAL_LM,
            lora_alpha=32,
            lora_dropout=0.05
        )
        self.peft_model = peft.get_peft_model(self.base_model, self.peft_config)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        inp = {"input_ids":batch["input_text"]['input_ids'], "attention_mask":batch["input_text"]['attention_mask']}
        logits = self.peft_model(**inp).loss["logits"].permute(0, 2, 1)

        out = batch["output_text"]['input_ids']
        loss = self.loss(logits, inp["input_ids"])
        return loss
    def predict(self, batch):
        inp = {"input_ids":batch["output_text"]['input_ids'], "attention_mask":batch["output_text"]['attention_mask']}
        with torch.no_grad():
            generated_ids = self.base_model.generate(**inp)
        out = {
            "out_ids":generated_ids
        }
        return out
    def show_output(self, batch):
        inp = {"input_ids":batch["output_text"]['input_ids'], "attention_mask":batch["output_text"]['attention_mask']}
        generated_ids = self.base_model.generate(**inp)
        generated_sentences = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("Generated sentence:")
        for s in generated_sentences:
            print(s)
        return