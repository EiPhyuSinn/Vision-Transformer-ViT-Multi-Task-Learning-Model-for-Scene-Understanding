# Vision Transformer (ViT) Multi-Task Learning Model for Scene Understanding

This repository contains a **Vision Transformer (ViT)-based multi-task learning model** for scene understanding using the **BDD10K dataset**. The model predicts **weather**, **scene type**, and **time of day** from images.

---

## Features

- **Dataset**: BDD10K, prepared and split for training and testing.
- **Implementation**: Converted from TensorFlow to **PyTorch**.
- (Ref : https://github.com/mnguyen0226/multitask_learning_vit/blob/main/multitask-learning-vit-cifar10-classification.ipynb)
- **Architecture**: Vision Transformer with multi-task heads for weather, scene, and time-of-day classification.
- **Training**: Supports ttraining up to 50 epochs with TensorBoard logging.
- **Inference**: Includes code for evaluating on test images.
- **Visualization**: Generates plots for loss and accuracy for each task.

---

## Model Overview

- **Patch Extraction**: Divides images into 6×6 patches.
- **Patch Encoding**: Linear projection + positional embedding.
- **Transformer Blocks**: Multi-head self-attention + feed-forward network.
- **Shared Features**: Flattened transformer output passed through an MLP.
- **Task-specific Heads**:
  - Weather (7 classes)
  - Scene (7 classes)
  - Time of Day (4 classes)
    
---

Example Results:

![00080_annotated](https://github.com/user-attachments/assets/f612821a-7db8-4805-bb8a-45540169f8e1)
![00104_annotated](https://github.com/user-attachments/assets/ebac38b9-e9d3-4044-b0fd-e7b7ea0d7bb3)


