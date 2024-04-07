import json
import argparse
from collections import defaultdict

import datasets
from tqdm import tqdm

import src.utils.data.formats as formats

FORMAT_TO_FUNC = {
    "raw": lambda messages, functions: json.dumps({"messages": messages, "functions": functions}),
    "llama": formats.llama_convert
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="data/dataset", help="Path to folder")
    parser.add_argument("-f", "--format", type=str, default="llama", help="Dataset format")
    args = parser.parse_args()

    if args.format not in FORMAT_TO_FUNC:
        raise ValueError(f"Format must be one of: {list(FORMAT_TO_FUNC.keys())}, got '{args.format}'")

    # Download raw hugginface dataset
    hf_dataset = datasets.load_dataset("korotkov/glaive-function-calling-v2-parsed")

    # Convert dataset to another format
    converted_dataset = datasets.DatasetDict()
    for part in ["train", "test"]:
        converted_dataset[part] = defaultdict(list)

        for row in tqdm(hf_dataset[part]):
            messages = json.loads(row["messages"])
            functions = json.loads(row["functions"])

            for feature, value in FORMAT_TO_FUNC[args.format](messages, functions).items():
                converted_dataset[part][feature].append(value)
        
        converted_dataset[part] = datasets.Dataset.from_dict(converted_dataset[part])

    # Save converted dataset
    converted_dataset.save_to_disk(args.path)
