# Multi-Stage Enhanced Zero Trust IDS

## Overview

This project implements a multi-stage zero trust intrusion detection system (IDS) for detecting unknown attacks in IoT and traditional networks. The system uses a novel three-stage architecture that combines supervised and unsupervised learning techniques to achieve high accuracy in detecting both known and unknown attacks.

## Key Features

- **Three-Stage Zero Trust Architecture**
  - Stage 1: Two layers of shallow deep learning classifiers for type-A detection
  - Stage 2: Two layers of deep learning models for type-B detection
  - Stage 3: DBSCAN clustering for unknown attack detection

- **Support for Multiple Datasets**
  - CIC-IDS-2017
  - CIC-IDS-2018
  - Bot-IoT
  - IoT-23

- **Zero Trust Security Model**
  - All traffic is considered malicious by default
  - Multiple verification layers
  - Comprehensive attack detection

## Getting Started

### Prerequisites

- Python 3.8+
- Virtual Environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/zerotrust-detection.git
cd zerotrust-detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Setup

1. Download one of the supported datasets:
   - CIC-IDS-2017: https://www.unb.ca/cic/datasets/ids-2017.html
   - CIC-IDS-2018: https://www.unb.ca/cic/datasets/ids-2018.html
   - Bot-IoT: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/
   - IoT-23: https://www.stratosphereips.org/datasets-iot23

2. Place the dataset files in the corresponding directory under `data/`

### Running the System

1. Train and evaluate on a dataset:
```bash
python src/main.py --dataset [dataset-name] --save-path models/[dataset-name]
```

2. Use a pre-trained model:
```bash
python src/main.py --dataset [dataset-name] --model-path models/[dataset-name]
```

## Project Structure

```
zerotrust-detection/
├── data/                    # Dataset directories
│   ├── cic-ids-2017/       # CIC-IDS-2017 dataset
│   ├── cic-ids-2018/       # CIC-IDS-2018 dataset
│   ├── bot-iot/            # Bot-IoT dataset
│   └── iot-23/            # IoT-23 dataset
├── models/                 # Saved models
├── results/                # Evaluation results
├── src/
│   ├── zero_trust_ids.py   # Core IDS implementation
│   ├── data_processor.py   # Data processing
│   └── main.py            # Main program
└── requirements.txt       # Project dependencies
```

## Results

The system outputs detailed metrics including:
- Classification accuracy
- Detection rate for known and unknown attacks
- Unknown attack ratio
- Confusion matrix
- Detailed classification report

Results are saved in the `results/` directory as JSON files.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset providers
- Research paper authors
- Open source community

## 📞 Contact

For questions and support, please open an issue in the GitHub repository. 