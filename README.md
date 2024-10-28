# NeurIPS-2024

## Overview
This repository contains the code and datasets for the paper submitted to NeurIPS 2024 titled **"Towards Effective Planning Strategies for Dynamic Opinion Networks."** The repository is organized into two main folders, `code` and `data`, to facilitate easy navigation and access to project components.

## Repository Structure

### `code`
This folder contains the implementation code for the project, divided into two main subfolders for Reinforcement Learning and Supervised Learning models:

- #### `RL`
  This subfolder contains the code for Reinforcement Learning (RL) models:
  - **ResNet/**: Code for training and testing the ResNet model on three different cases (Case1, Case2, Case3) with five reward functions (R_0 to R_4).
  - **GCN/**: Code for training and testing the GCN model on three different cases (Case1, Case2, Case3) with six reward functions (R_0 to R_5).

- #### `SL-GCN`
  This subfolder contains the Supervised Learning (SL) models and associated scripts, specifically for GCN-based supervised learning tasks.

### `data`
This folder contains the datasets used for testing and evaluation of the models. It includes the following subfolders:

- **Dataset_v1**: Contains datasets with varying initial numbers of infected nodes.
- **Dataset_v2**: Contains datasets with varying degrees of connectivity for the initial infected nodes.

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{
  muppasani2024towards,
  title={Towards Effective Planning Strategies for Dynamic Opinion Networks},
  author={Bharath Muppasani, Protik Nag, Vignesh Narayanan, Biplav Srivastava, and Michael N. Huhns},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=LYivxMp5es}
}
