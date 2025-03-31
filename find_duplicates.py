"""
This script loads a JSON dataset (Freiburg Dataset Format having a list of dictionaries),
identifies and removes duplicate entries based on the "rgb_path" field, 
and overwrites the original file with the cleaned data.
"""
import json
from collections import defaultdict

def deduplicate_by_rgb_path(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"Original dataset length: {len(data)}")

    seen = {}
    duplicates = []

    cleaned_data = []
    for idx, entry in enumerate(data):
        rgb = entry["rgb_path"]
        if rgb in seen:
            duplicates.append((rgb, seen[rgb], idx))
        else:
            seen[rgb] = idx
            cleaned_data.append(entry)

    if duplicates:
        print(f"Found {len(duplicates)} duplicates:")
        for rgb_path, first_idx, dup_idx in duplicates:
            print(f"\nRGB Path: {rgb_path}\n  First index: {first_idx}, Duplicate index: {dup_idx}")
    else:
        print("No duplicates found.")

    print(f"\n New dataset length after removing duplicates: {len(cleaned_data)}")

    # Overwrite the JSON file
    with open(json_path, "w") as f:
        json.dump(cleaned_data, f, indent=4)
    print(f"Cleaned JSON saved to: {json_path}")

if __name__ == "__main__":
    dataset_path = "./dataset_v1_224.json"
    deduplicate_by_rgb_path(dataset_path)
