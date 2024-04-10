import json

import wandb
import torch
from tqdm import tqdm
import transformers
from transformers.integrations import WandbCallback


def get_tool_call(message: str, tool_call_b, tool_call_e) -> dict:
    if tool_call_b in message:
        l = message.find(tool_call_b) + len(tool_call_b)
        r = l + message[l:].find(tool_call_e)
        if r < l:
            return {}

        return json.loads(message[l:r].strip())
    else:
        return {}


class LLMMetricsCallback(WandbCallback):
    def __init__(
        self,
        trainer,
        dataset,
        tool_call_b,
        tool_call_e,
        max_new_tokens=256,
        log_model="checkpoint",
    ):
        super().__init__()
        self._log_model = log_model
        self.dataset = dataset
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.tool_call_b, self.tool_call_e = tool_call_b, tool_call_e

        self.gen_config = transformers.GenerationConfig.from_pretrained(
            trainer.model.name_or_path,
            max_new_tokens=max_new_tokens,
        )

    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

        with torch.inference_mode():
            output = self.model.generate(tokenized_prompt, generation_config=self.gen_config)

        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=False)

    def samples_table(self, examples):
        records_table = wandb.Table(columns=["prompt", "generation", "target"] + list(self.gen_config.to_dict().keys()))

        metrics = {
            "valid_json_call": len(examples),
            "correct_call_name": 0,
            "correct_call_args": 0,
        }

        for example in tqdm(examples, leave=False):
            print(example.keys())
            prompt = example["text"]
            target = example["text_target"]
            generation = self.generate(prompt=prompt)


            # Add row to table
            records_table.add_data(prompt, generation, target, *list(self.gen_config.to_dict().values()))
            
            # Calculate metrics
            target_function_call = get_tool_call(target, self.tool_call_b, self.tool_call_e)
            try:
                generation_function_call = get_tool_call(generation, self.tool_call_b, self.tool_call_e)
            except json.decoder.JSONDecodeError as e:
                metrics["valid_json_call"] -= 1
                generation_function_call = {}
                continue

            metrics["correct_call_name"] += target_function_call.get("name") == generation_function_call.get("name")
            metrics["correct_call_args"] += target_function_call.get("arguments") == generation_function_call.get("arguments")

        for metric in metrics:
            metrics[metric] /= len(examples)

        return records_table, metrics
        
    def on_evaluate(self, args, state, control,  **kwargs):
        super().on_evaluate(args, state, control, **kwargs)

        # Log samples
        records_table, metrics = self.samples_table(self.dataset)
        self._wandb.log({"sample_predictions": records_table})

        # Log metrics
        self._wandb.log(metrics)
