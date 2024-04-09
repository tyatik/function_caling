import json
from collections import defaultdict

import datasets
from datasets import load_dataset
from tqdm import tqdm

from .llama import convert as llama_convert
from .llama_ru_translated import convert as llama_ru_translated_convert
from .preprocessors import korotkov_preprocessor

FORMAT_TO_FUNC = {
    "raw": lambda messages, functions: json.dumps({"messages": messages, "functions": functions}),
    "llama": llama_convert,
    "llama_ru_translated": llama_ru_translated_convert
}

DATASET_TO_PREPROCESSOR = {
    "korotkov/glaive-function-calling-v2-parsed":korotkov_preprocessor
}


def convert_dataset(dataset: str, format: str, push_frequency: int = 0, push_repository: str = "", push_token: str = ""):
    if format not in FORMAT_TO_FUNC:
        raise ValueError(f"Format must be one of: {list(FORMAT_TO_FUNC.keys())}, got '{format}'")
    if dataset not in DATASET_TO_PREPROCESSOR:
        raise ValueError(f"Dataset must be one of: {list(DATASET_TO_PREPROCESSOR.keys())}, got '{format}'")
    
    #Load and preprocess dataset
    hf_dataset = load_dataset(dataset)
    dataset = DATASET_TO_PREPROCESSOR[dataset](hf_dataset)
    del hf_dataset

    # Convert dataset to another format
    buffer_dataset = datasets.DatasetDict()
    converted_dataset = datasets.DatasetDict()
    for part in ["train", "test"]:
        buffer_dataset[part] = defaultdict(list)
        converted_dataset[part] = defaultdict(list)
        messages = dataset[part]["messages"]
        functions = dataset[part]["functions"]

        if push_frequency == 0:
            push_frequency == len(messages)

        step = 0
        for i in range(len(messages)):
            r = (step*push_frequency, step*push_frequency+push_frequency)
            if step*push_frequency+push_frequency > len(messages):
                r[1] = len(messages)
            local_messages = messages[r[0]: r[1]]
            local_functions = functions[r[0]: r[1]]

            for feature, value in FORMAT_TO_FUNC[format](local_messages, local_functions).items():
                buffer_dataset[part][feature].extend(value)

            converted_dataset[part] = datasets.Dataset.from_dict(buffer_dataset[part])
            if push_repository != "" and push_token != "":
                converted_dataset.push_to_hub(push_repository, token=push_token)

            if r[1] == len(messages):
                break
            step += 1

    return converted_dataset