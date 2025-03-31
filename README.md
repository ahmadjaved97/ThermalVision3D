# ThermalVision3D
Final Project for MA-INF-4308: CUDA Vision Lab

This repository contains the file and links for all the material used to train the Dust3r model on thermal images.

### 1. Dust3r Repository Used: https://github.com/ahmadjaved97/dust3r
This Dust3r repository contains the freiburg data loader class inside `dust3r/dust3r/dataset` used to train the model.

### 2. Mast3r Repository Used: https://github.com/ahmadjaved97/mast3r
Any changes made to the Mast3r reposity are present here.

### 3. Pseudo GT Visualization `(3D Camera Transformations and Visualizations.ipynb)`
All the visualization and verifcation for pseudo gt is present in this jupyter notebook.

#### 4. Dataset Evaluation `(evaluate.py)` 
This file was used to evalute the dust3r model on the test set.

#### 5. Result Visualization
1. `test_model_freiburg.py`: used to generate results on the freiburg dataset using the best model and the baseline model.
2. `test_model_ais.py`: used to generate results on the AIS(FLIR_BOSON) dataset
3. `AIS Inference Visualization.ipynb`: Visualized results from the AIS dataset.
4. `Freiburg Test Data Visualization.ipynb`: Visualized results from the Freiburg Dataset using the best model and the baseline model.

#### 6. Tensorboard Logs
All the tensorboard logs are present inside `tensorboard_logs` folder for the experiments mentioned in the report.

#### 7. Dataset Splitter `(split_json.py)` 
Splits a JSON dataset into train and test sets based on a user-defined ratio, preserving order or using optional shuffling.

#### 8. Deduplicator by RGB Path  `(find_duplicates.py)`
Removes duplicate entries from a JSON dataset based on the `rgb_path` field and overwrites the original file with the cleaned data.
