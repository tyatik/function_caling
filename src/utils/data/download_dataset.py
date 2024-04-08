import sys
sys.path.append("./")

import argparse
import datasets

from src.utils.data.formats import convert_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="data/dataset", help="Path to folder")
    parser.add_argument("-f", "--format", type=str, default="llama", help="Dataset format")
    args = parser.parse_args()

    # Download raw huggingface dataset
    hf_dataset = datasets.load_dataset("korotkov/glaive-function-calling-v2-parsed")

    # Convert dataset
    converted_dataset = convert_dataset(
        dataset=hf_dataset,
        format=args.format,
    )

    # Save dataset to disk
    converted_dataset.save_to_disk(args.path)
