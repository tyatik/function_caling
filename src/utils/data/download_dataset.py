import sys
sys.path.append("./")

import argparse
import datasets

from formats import convert_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="korotkov/glaive-function-calling-v2-parsed", help="Dataset to load")
    parser.add_argument("-f", "--format", type=str, default="llama", help="Dataset format")
    parser.add_argument("-p", "--path", type=str, default="data/dataset", help="Path to folder")
    parser.add_argument("-n", "--push_n", type=int, default=0, help="Push to hub every n samples")
    parser.add_argument("-r", "--push_r", type=str, default="", help="Repository to push")
    parser.add_argument("-t", "--push_t", type=str, default="", help="Token to push")
    
    args = parser.parse_args()

    # Convert dataset
    converted_dataset = convert_dataset(
        dataset=args.dataset,
        format=args.format,
        push_frequency=args.push_n,
        push_repository=args.push_r,
        push_token=args.push_t
    )

    # Save dataset to disk
    converted_dataset.save_to_disk(args.path)
