import json
from collections import defaultdict

import datasets
from tqdm import tqdm

from .llama import convert as llama_convert
from .llama_ru import convert as llama_ru_convert

FORMAT_TO_FUNC = {
    "raw": lambda messages, functions: json.dumps({"messages": messages, "functions": functions}),
    "llama": llama_convert,
    "llama_ru": llama_ru_convert
}


def convert_dataset(dataset, format: str, push_n: int = 100):
    if format not in FORMAT_TO_FUNC:
        raise ValueError(f"Format must be one of: {list(FORMAT_TO_FUNC.keys())}, got '{format}'")

    # Convert dataset to another format
    buffer_dataset = datasets.DatasetDict()
    converted_dataset = datasets.DatasetDict()
    for part in ["train", "test"]:
        buffer_dataset[part] = defaultdict(list)
        converted_dataset[part] = defaultdict(list)
        messages = dataset[part]["messages"]
        functions = dataset[part]["functions"]

        step = 0
        for i in range(len(dataset[part]["messages"])):
            r = (step*push_n, step*push_n+push_n)
            if step*push_n+push_n > len(dataset[part]["messages"]):
                r[1] = len(dataset[part]["messages"])
            local_messages = dataset[part]["messages"][r[0]: r[1]]
            local_functions = dataset[part]["functions"][r[0]: r[1]]

            for feature, value in FORMAT_TO_FUNC[format](local_messages, local_functions).items():
                buffer_dataset[part][feature].extend(value)

            converted_dataset[part] = datasets.Dataset.from_dict(buffer_dataset[part])
            converted_dataset.push_to_hub("evgmaslov/glaive-function-calling-v2-parsed-ru", token="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")

            if r[1] == len(dataset[part]["messages"]):
                break
            step += 1

    return converted_dataset