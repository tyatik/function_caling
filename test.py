import yaml
import argparse
import warnings
import random
from collections import defaultdict

import torch
import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    StoppingCriteriaList,
)
from peft import PeftModel

from src.utils.data.formats import FORMATS_DICT
from src.utils.stopping_criterias import KeywordsStoppingCriteria
from src.utils.metrics import generate, calculate_metrics


def generate(model, tokenizer, generation_config, stopping_criteria_list, prompt):
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

    with torch.inference_mode():
        output = model.generate(
            tokenized_prompt,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria_list,
        )

    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/test.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config.get("filter_warnings", False):
        warnings.filterwarnings("ignore")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    quantization_config = BitsAndBytesConfig(**config["model"]["quantization_config_args"])
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model_name"],
        device_map=config["model"]["device"],
        quantization_config=quantization_config
    )
    model = PeftModel.from_pretrained(
        model=base_model,
        model_id=config["model"]["adapter_model_name"],
        device_map=config["model"]["device"],
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_name"], **config["model"]["tokenizer_args"])
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("Loading data...")
    dataset = datasets.load_dataset(config["data"]["name"])

    # Sample test data
    random.seed(42)
    samples = dataset["test"].select(
        random.sample(
            range(len(dataset["test"])),
            config["data"]["num_samples"]),
    )
    samples_formatted = defaultdict(list)
    for sample in samples:
        for key, value in FORMATS_DICT[config["data"]["format"]]["test"](sample).items():
            samples_formatted[key].extend(value)
    samples_formatted = datasets.Dataset.from_dict(samples_formatted)

    # Create generation config and stopping criteria
    generation_config = GenerationConfig.from_pretrained(
        config["model"]["base_model_name"],
        max_new_tokens=config["generation"]["max_new_tokens"],
    )
    stopping_criteria_list = StoppingCriteriaList([
        KeywordsStoppingCriteria(tokenizer=tokenizer, stopwords=config["generation"]["stopwords"])
    ])

    # Calcualate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(
        samples_formatted,
        format=config["data"]["format"],
        generate_func=lambda prompt: generate(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            stopping_criteria_list=stopping_criteria_list,
        ),
    )

    for metric in metrics:
        print(f"{metric:18}: {metrics[metric]:.5f}")
