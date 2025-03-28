"""
This script loads a JSON dataset (a list of dictionaries) from a specified file,
splits it while preserving the order (not randomized) into training and testing subsets based on a user-defined ratio,
and saves the resulting files in the same directory as the original.
"""
import os
import json
from rich import print

def split_dataset(input_file, train_ratio=0.8):
    # Load the dataset
    with open(input_file, 'r') as f:
        data = json.load(f)

    total_len = len(data)

    split_index = int(total_len * train_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    base_dir = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    train_file = os.path.join(base_dir, f"{base_name}_train.json")
    test_file = os.path.join(base_dir, f"{base_name}_test.json")

    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)

    return {
        "total_samples": total_len,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "train_file_path": train_file,
        "test_file_path": test_file
    }

if __name__ == "__main__":
    dataset_path = "/lustre/mlnvme/data/s63ajave_hpc-cuda_lab/dataset_v1_224.json"
    result = split_dataset(dataset_path, train_ratio=0.95)
    print(result)
