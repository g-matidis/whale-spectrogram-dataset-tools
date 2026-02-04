# 🐋 whale-spectrogram-dataset-tools

A PyTorch-based toolkit for loading, processing, visualizing, and evaluating Humpback Whale song spectrograms. This repository serves as supplementary material to the Humpback Whales Spectrogram Dataset.

## 📂 Project Structure

- `src/whales_dataset.py`: Custom PyTorch `Dataset` classes (`LineLevelDataset`, `PageLevelDataset`) that handle complex JSON annotations.
- `src/transforms.py`: Specialized operations on the data, including patches creation (`RandomSpectrogramLinePatcher`) and format converters (`UnitIntervalsToYOLO`).
- `src/evaluate.py`: Evaluation script to calculate Precision, Recall, and mAP for object detection.
- `src/visualization.py`: Script for visualizing bounding boxes/polygons.
- `src/utils.py`: General helper functions (e.g. `is_valid_file`) and parsers (e.g. `parse_line_level_data`).

## 🚀 Installation

This project assumes that you 

1. Clone the repository:
   ```bash
   git clone https://github.com/g-matidis/whale-spectrogram-dataset-tools.git
   cd whale-spectrogram-dataset-tools
2. 