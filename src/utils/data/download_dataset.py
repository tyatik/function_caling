import json
import argparse

from datasets import load_dataset
from tqdm import tqdm

import src.utils.data.formats as formats

FORMAT_TO_FUNC = {
    "llama": formats.llama_convert
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="data/dataset.json", help="Path to folder")
    parser.add_argument("-f", "--format", type=str, default="llama", help="Dataset format")
    args = parser.parse_args()

    if args.format not in FORMAT_TO_FUNC:
        raise ValueError(f"Format must be one of: {list(FORMAT_TO_FUNC.keys())}, got '{args.format}'")

    # Download raw hugginface dataset
    hf_dataset = load_dataset("korotkov/glaive-function-calling-v2-parsed")

    # Convert dataset to another format
    dataset = {"train": [], "test": []}
    for part in ["train", "test"]:
        for row in tqdm(hf_dataset[part]):
            messages = json.loads(row["messages"])
            functions = json.loads(row["functions"])

            dataset[part].append(FORMAT_TO_FUNC[args.format](messages, functions))

    # Save converted dataset as JSON file
    with open(args.path, "w+") as f:
        json.dump(dataset, f)
