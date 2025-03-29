# ThermalVision3D
Final Project for MA-INF-4308: CUDA Vision Lab

#### Dataset Splitter `(split_json.py)` 
Splits a JSON dataset into train and test sets based on a user-defined ratio, preserving order or using optional shuffling.

#### Deduplicator by RGB Path  `(find_duplicates.py)`
Removes duplicate entries from a JSON dataset based on the `rgb_path` field and overwrites the original file with the cleaned data.
