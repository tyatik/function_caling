import yaml
import argparse

import datasets
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

from src.utils.data.formats import convert_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/train.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(config["model"]["name"], device_map=config["model"]["device"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], **config["model"]["tokenizer_args"])
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("Loading data...")
    hf_dataset = datasets.load_dataset(config["data"]["name"])
    dataset = convert_dataset(
        dataset=hf_dataset,
        format=config["data"]["format"],
    )
    del hf_dataset
    dataset = dataset.map(
        lambda row: tokenizer(row["text"], **config["data"]["tokenizer_args"]),
        batched=True,
    )
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Init trainer
    lora_config = LoraConfig(**config["model"]["lora_args"])

    model.train()
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    training_args = transformers.TrainingArguments(**config["training_arguments"])
    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        data_collator=data_collator,
    )

    # Train model
    print("Model training...")
    trainer.train()
