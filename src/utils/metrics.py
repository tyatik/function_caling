from typing import Dict, Callable
import json

import torch
from tqdm import tqdm
from transformers import GenerationConfig, StoppingCriteriaList

from src.utils.data.formats import FORMATS_DICT


def generate(
    prompt: str,
    model,
    tokenizer,
    generation_config: GenerationConfig,
    stopping_criteria_list: StoppingCriteriaList,
) -> str:
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

    with torch.inference_mode():
        output = model.generate(
            tokenized_prompt,
            generation_config=generation_config,
            stopping_criteria_list=stopping_criteria_list,
        )

    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=False)


def get_tool_call(message: str, tool_call_b: str, tool_call_e: str) -> dict:
    if tool_call_b in message:
        l = message.find(tool_call_b) + len(tool_call_b)
        r = l + message[l:].find(tool_call_e)
        if r < l:
            return {}

        return json.loads(message[l:r].strip())
    else:
        return {}


def valid_json_call(generation: str, tool_call_b: str, tool_call_e: str) -> bool:
    try:
        get_tool_call(generation, tool_call_b, tool_call_e)
        return True
    except json.decoder.JSONDecodeError as e:
        return False


def correct_call_name(target: dict, generation: dict) -> bool:
    return target.get("name") == generation.get("name")


def correct_call_args(target: dict, generation: dict) -> bool:
    return target.get("args") == generation.get("args")


def calculate_metrics(
    dataset,
    generate_func: Callable,
    format: str,
) -> Dict[str, bool]:
    metrics = {
        "valid_json_call": 0,
        "correct_call_name": 0,
        "correct_call_args": 0,
    }

    tool_call_b = FORMATS_DICT[format]["tool_call_b"]
    tool_call_e = FORMATS_DICT[format]["tool_call_e"]
    for example in tqdm(dataset, leave=False):
        prompt = example["text"]
        target = example["text_target"]
        generation = generate_func(prompt=prompt)

        # Calculate metrics
        target_function_call = get_tool_call(target, tool_call_b, tool_call_e)
        generation_function_call = get_tool_call(generation, tool_call_b, tool_call_e)

        if valid_json_call(generation, tool_call_b, tool_call_e):
            metrics["valid_json_call"] += 1
        if correct_call_name(target_function_call, generation_function_call):
            metrics["correct_call_name"] += 1
        if correct_call_args(target_function_call, generation_function_call):
            metrics["correct_call_args"] += 1

    for metric in metrics:
        metrics[metric] /= len(dataset)
    
    return metrics
