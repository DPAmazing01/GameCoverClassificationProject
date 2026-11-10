(Project was worked on from March-April 2026 in a team of 3, where I worked on defining the custom ResNet18 model and the training/testing loop on the custom model)

## Project Overview

This project trains a convolutional neural network (custom ResNet-18) to predict a game's genre(s) based on its cover image. The model uses the Steam Games 2025 Kaggle dataset and handles a 33-class multi-label problem. It aims to improve upon existing baselines such as GameNet-1.

## Results

- Micro F1: 0.578
- Macro F1: 0.100
- Element-wise accuracy: 92.8%

The high element-wise accuracy is due to severe class imbalance. The model performs well on frequent genres (Indie, Action, Adventure, Casual) but misses most of the rare ones. Detailed per-genre metrics and confusion matrices are available in the notebook.

## Dataset

The dataset is **Steam Games 2025**, available on Kaggle:  
[Steam Games Dataset 2025](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)

In particular, the `games_march2025_cleaned.csv` file from the Kaggle dataset was used, and the notebook extracts the images contained in the CSV file into a folder called `steam_images`. 

## Model Architecture 

A custom ResNet-18 implemented from scratch (no pretrained weights).

- Input: 224x224 RGB images, normalized with ImageNet statistics.
- Four residual layers with BasicBlocks: 64, 128, 256, 512 output channels.
- Global average pooling followed by a fully-connected layer (512 to 33).
- Multi-label outputs trained with Binary Cross-Entropy with Logits loss.
- Adam optimizer (lr=1e-4), batch size 8.

During training, the model state is saved whenever validation loss improves to `best_resnet18_steam.pth`.

## Evaluation

The test set (30% hold-out) is evaluated using:

- Per-genre precision, recall, F1-score.
- Per-genre confusion matrices (TP, FP, FN, TN).
- Micro and macro averaged F1 scores.

All metrics are reported in the notebook.

## Credits

The Steam Games 2025 dataset is maintained by Artermiloff on Kaggle. The GameNet-1 model provided baseline motivation (Link to the baseline model: https://huggingface.co/MAS-AI-0000/GameNet-1).
