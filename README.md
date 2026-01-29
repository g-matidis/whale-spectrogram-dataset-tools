# 🐋 whale-spectrogram-dataset-tools

A PyTorch-based toolkit for loading, processing, visualizing, and evaluating Humpback Whale song spectrograms. This repository serves as supplementary material to the Humpback Whales Spectrogram Dataset.

## 📂 Project Structure

- `src/whales_dataset.py`: Custom PyTorch `Dataset` classes (`LineLevelDataset`, `PageLevelDataset`) that handle complex JSON annotations.
- `src/transforms.py`: Specialized augmentations (e.g., `RandomSpectrogramLinePatcher`) and format converters (intervals → YOLO).
- `src/visualization.py`: CLI tool for debugging and verifying bounding boxes/polygons.
- `src/evaluate.py`: Evaluation script to calculate Precision, Recall, and mAP for object detection.
- `src/utils.py`: Shared JSON parsing logic.

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/g-matidis/whale-spectrogram-dataset-tools.git](https://github.com/g-matidis/whale-spectrogram-dataset-tools.git)
   cd whale-spectrogram-dataset-tools