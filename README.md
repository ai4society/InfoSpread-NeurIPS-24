# NeurIPS-Submission-2024
## Overview

This repository contains the code and data for the paper submitted to NeurIPS 2024 titled "Towards Effective Planning Strategies for Dynamic Opinion Networks" The repository is structured into two main folders: `code` and `data`.


## Folders Description

### code/

This folder contains the implementation code for the project. It is divided into two main subfolders:

#### RL/
Contains the code for Reinforcement Learning models. This includes:

- **ResNet/**: Contains training and testing code for the ResNet model for three different cases (Case1, Case2, Case3) with five reward functions (R_0 to R_4).
- **GCN/**: Contains training and testing code for the GCN model for three different cases (Case1, Case2, Case3) with six reward functions (R_0 to R_5).

#### SL-GCN/
This folder is for Supervised Learning models and associated scripts.

### data/

This folder contains the datasets used for testing. It is divided into two subfolders:

- **Dataset_v1/**: Contains the dataset files with varying initial infected nodes.
- **Dataset_v2/**: Contains the dataset files with varying Degrees of Connectivity for the initial infected nodes.

