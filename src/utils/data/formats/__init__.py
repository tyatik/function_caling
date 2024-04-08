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


def convert_dataset(dataset, format: str):
    if format not in FORMAT_TO_FUNC:
        raise ValueError(f"Format must be one of: {list(FORMAT_TO_FUNC.keys())}, got '{format}'")

    # Convert dataset to another format
    converted_dataset = datasets.DatasetDict()
    for part in ["train", "test"]:
        converted_dataset[part] = defaultdict(list)
        messages = dataset[part]["messages"][:10]
        functions = dataset[part]["functions"][:10]

        for feature, value in FORMAT_TO_FUNC[format](messages, functions).items():
            converted_dataset[part][feature] = value
        
        converted_dataset[part] = datasets.Dataset.from_dict(converted_dataset[part])
    
    return converted_dataset