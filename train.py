import yaml
import argparse
import warnings
import random
from collections import defaultdict

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training, LoraConfig

from src.utils.data.formats import FORMATS_DICT
from src.callbacks.metrics import LLMMetricsCallback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/train.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config.get("filter_warnings", False):
        warnings.filterwarnings("ignore")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(config["model"]["name"], device_map=config["model"]["device"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], **config["model"]["tokenizer_args"])
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("Loading data...")
    dataset = datasets.load_dataset(config["data"]["name"])

    # Loss eval samples
    dataset["test"] = dataset["test"].select(
        random.sample(
            range(len(dataset["test"])),
            config["data"]["loss_eval_samples"]),
    )

    # Metrics eval samples
    metrics_samples = dataset["test"].select(
        random.sample(
            range(len(dataset["test"])),
            config["data"]["metrics_eval_samples"]),
    )
    metrics_samples_formatted = defaultdict(list)
    for sample in metrics_samples:
        for key, value in FORMATS_DICT[config["data"]["format"]]["test"](sample).items():
            metrics_samples_formatted[key].extend(value)
    metrics_samples_formatted = datasets.Dataset.from_dict(metrics_samples_formatted)

    # Init trainer
    lora_config = LoraConfig(**config["model"]["lora_args"])

    model.train()
    model = prepare_model_for_kbit_training(model)

    training_args = TrainingArguments(**config["training_args"])
    trainer = SFTTrainer(
        model=model,
        peft_config=lora_config,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=FORMATS_DICT[config["data"]["format"]]["train"],
        **config["sft_trainer_args"],
    )

    # Add metrics callback
    wandb_callback = LLMMetricsCallback(
        trainer,
        metrics_samples_formatted,
        tool_call_b=FORMATS_DICT[config["data"]["format"]]["tool_call_b"],
        tool_call_e=FORMATS_DICT[config["data"]["format"]]["tool_call_e"],
        **config["metrics_callback"],
    )
    trainer.add_callback(wandb_callback)

    # Train model
    print("Model training...")
    trainer.train()
