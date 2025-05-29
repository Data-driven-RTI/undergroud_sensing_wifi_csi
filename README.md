# Underground Root Tuber Sensing via Wi-Fi Mesh Network

This repository contains the code, data, and documentation for the project described in the paper **"Underground Root Tuber Sensing via a Wi-Fi Mesh Network"**, presented at the 23rd ACM Conference on Embedded Networked Sensor Systems (SenSys '25). The project demonstrates a non-invasive Wi-Fi sensing system that uses channel state information (CSI) data and deep neural network (DNN) models to reconstruct cross-sectional images of potato tubers underground.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Description](#dataset-description)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project leverages Wi-Fi CSI data and a multi-branch convolutional neural network (CNN) model to reconstruct high-resolution images of underground potato tubers. The system comprises:
1. A Wi-Fi mesh network for CSI data acquisition.
2. A deep neural network model for image reconstruction.

The dataset includes CSI measurements and ground truth masks for 26 potato tubers, collected using a custom-built testbed.

---

## Repository Structure

The repository is organized as follows:

```
neural_netwok/
    csi_dataset.py    # PyTorch dataset for CSI data
    model.py          # Multi-branch CNN model
    rssi_dataset.py   # PyTorch dataset for RSSI data
processed_data_csi/
    potato_1/         # Processed CSI data for potato 1
    potato_2/         # Processed CSI data for potato 2
    ...
processed_data_rssi/
    potato_1/         # Processed RSSI data for potato 1
    potato_2/         # Processed RSSI data for potato 2
    ...
raw_data/
    calibration/      # Calibration data
    potato_1/         # Raw data for potato 1
    potato_1_gt/      # Ground truth for potato 1
    ...
scripts/
    compare_all_models.py  # Script to compare models
    train_model.py         # Script to train the model
    run_all_experiments.py # Script to run experiments
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- PyTorch 2.0 or higher
- Additional Python libraries: `numpy`, `pandas`, `matplotlib`, `argparse`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Data-driven-RTI/undergroud_sensing_wifi_csi
   cd undergroud_sensing_wifi_csi
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Dataset Preparation
1. **Raw Data**: The `raw_data/` directory contains calibration data and ground truth masks for each potato tuber.
2. **Processed Data**: The `processed_data_csi/` and `processed_data_rssi/` directories contain CSI and RSSI features, respectively.

### Training the Model
To train the multi-branch CNN model:
```bash
python scripts/train_model.py --data_dir processed_data_csi --model_type unet --epochs 50
```

### Running Experiments
To run all experiments:
```bash
python scripts/run_all_experiments.py --config config.json
```

### Comparing Models
To compare different models:
```bash
python scripts/compare_all_models.py --data_dir processed_data_csi
```

---

## Dataset Description

### CSI Data
- **Processed CSI Data**: Stored in `processed_data_csi/`, organized by potato tuber (`potato_1`, `potato_2`, etc.).
- **Features**: CSI tensors with dimensions `(156, 12, 12)`.

### RSSI Data
- **Processed RSSI Data**: Stored in `processed_data_rssi/`, organized similarly to CSI data.
- **Features**: RSSI tensors with dimensions `(3, 16, 16)`.

### Ground Truth
- **Raw Data**: Ground truth masks are stored in `raw_data/` under `potato_1_gt`, `potato_2_gt`, etc.

---

## Citation

If you use this repository in your research, please cite our paper:

```
@inproceedings{Elhadi2025WiFiSensing,
  author = {Said Elhadi and Tao Wang and Yang Zhao},
  title = {Underground Root Tuber Sensing via a Wi-Fi Mesh Network},
  year = {2025},
  booktitle = {Proceedings of the 23rd ACM Conference on Embedded Networked Sensor Systems (SenSys '25)},
  doi = {10.1145/3715014.3724365}
}
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
