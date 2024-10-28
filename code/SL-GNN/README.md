# NeurIPS-Submission-2024

## Overview

This repository contains all the necessary scripts, models, and data to reproduce the experiments and results presented in our NeurIPS 2024 submission. The project focuses on the comparison of different models in the context of misinformation spread in dynamical systems.

## Folder Structure

- **Comparison/**
  - Contains CSV files with the results of model comparisons for different cases.

- **Figures/**
  - Contains the generated figures from the comparison results.

- **Models/**
  - Contains the weight files of the models used for comparisons across different cases.

## File Structure

- **`comparison_dataset_v1.py`**
  - Compares different models on Dataset V1.

- **`comparison_general.py`**
  - Compares different models on Dataset V2.

- **`figures.py`**
  - Generates figures from the comparison results.

- **`GNN.py`**
  - Defines the Graph Convolutional Network (GCN) model.

- **`graph_utils.py`**
    - Updating graph environment
    - Fetching graph properties
    - Generate new graph states 
    - Simulating the spread of misinformation
    - Implementation of Ranking Algorithm

- **`main.py`**
  - Trains the GNN models for 3 different cases (case 1, 2 and 3)

- **`MisInfoSpread.py`**
  - Defines the placeholder MisInfoSpread model, essential for using the dataset.

- **`model_utils.py`**
  - Defines various model utilities such as selecting the top-k nodes.

- **`params.py`**
  - Contains the parameters used throughout the project.

- **`visualization_utils.py`**
  - Provides visualization utilities, including graph plotting for debugging purposes.

## Usage Instructions

### Training Models

To train the different models, execute the `main.py` script. Model parameters can be adjusted in the `params.py` file.

```bash
python main.py
```

### Comparing Models

To compare the models on different datasets, run the comparison scripts.

```bash
python comparison_dataset_v1.py
python comparison_general.py
```

### Generating Figures

To generate figures from the comparison results, execute the `figures.py` script.

```bash
python figures.py
```