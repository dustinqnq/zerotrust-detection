# Multi-Stage Zero Trust Intrusion Detection System

## 🌟 Overview

This project implements a sophisticated multi-stage zero trust intrusion detection system (IDS) that combines multiple machine learning algorithms to detect and classify IoT-based threats and unknown attacks. The system uses a novel three-stage architecture that combines supervised and unsupervised learning techniques to achieve high accuracy in detecting both known and unknown attacks.

## 🎯 Key Features

### Three-Stage Zero Trust Architecture
- **Stage 1**: Two layers of shallow deep learning classifiers for type-A detection
- **Stage 2**: Two layers of deep learning models for type-B detection  
- **Stage 3**: DBSCAN clustering for unknown attack detection

### Advanced Features
- **Multi-Dataset Support**: CIC-IDS-2017, CIC-IDS-2018, Bot-IoT, IoT-23
- **Zero Trust Security Model**: All traffic considered malicious by default
- **Dynamic Processing**: Intelligent feature extraction and preprocessing
- **Real-time Detection**: Efficient processing with gradient stabilization

### Performance Improvements
- **Gradient Explosion Prevention**: BatchNormalization and optimized learning rates
- **Early Stopping**: Prevents overfitting with validation monitoring
- **Dynamic Label Handling**: Automatic dimension matching for different datasets

## 📊 Performance Metrics

- High classification accuracy on multiple datasets
- Effective detection of both known and unknown attacks
- Low false positive rates
- Real-time processing capabilities

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Virtual Environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dustinqnq/zerotrust-detection.git
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
   - **IoT-23**: https://www.stratosphereips.org/datasets-iot23 (Currently supported)
   - CIC-IDS-2017: https://www.unb.ca/cic/datasets/ids-2017.html
   - CIC-IDS-2018: https://www.unb.ca/cic/datasets/ids-2018.html
   - Bot-IoT: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/

2. Place the dataset files in the corresponding directory under `data/`

### Running the System

1. **Test with IoT-23 dataset** (recommended):
```bash
python src/test_iot23.py
```

2. **Download IoT-23 data** (if needed):
```bash
python src/download_dataset.py
```

3. **Train custom model**:
```bash
python src/main.py --dataset [dataset-name] --save-path models/[dataset-name]
```

## 📁 Project Structure

```
zerotrust-detection/
├── src/
│   ├── processors/                 # Data processing modules
│   │   ├── base_processor.py      # Base data processor
│   │   └── iot23_processor.py     # IoT-23 specific processor
│   ├── zero_trust_ids.py          # Core IDS implementation
│   ├── test_iot23.py             # IoT-23 testing script
│   ├── download_dataset.py       # Dataset download utility
│   ├── data_processor.py         # General data processing
│   └── main.py                   # Main program
├── data/                          # Dataset directories (ignored)
├── models/                        # Saved models (ignored)
├── results/                       # Evaluation results
├── requirements.txt              # Project dependencies
├── PROGRESS.md                   # Development progress
└── .gitignore                    # Git ignore rules
```

## 🔧 Recent Improvements

### Code Organization
- **Unified Data Processors**: All data format converters organized in `src/processors/` package
- **Base Class Architecture**: Extensible `BaseDataProcessor` for adding new datasets
- **Modular Design**: Clean separation of concerns and easy maintenance

### Model Enhancements
- **Gradient Stabilization**: Fixed gradient explosion with BatchNormalization and learning rate optimization
- **Dynamic Architecture**: Models adapt to actual data dimensions automatically
- **Training Improvements**: Added EarlyStopping and validation monitoring

### Current Dataset Support
- **IoT-23**: Fully implemented with Zeek/Bro log parsing
- **Feature Engineering**: 27 extracted features including protocol, service, and statistical features
- **Label Processing**: Support for both binary and multi-class classification

## 📈 Results and Evaluation

The system outputs detailed metrics including:
- Classification accuracy for each stage
- Detection rate for known and unknown attacks
- DBSCAN clustering analysis
- Detailed performance reports

Results are automatically saved in the `models/` directory.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset providers (IoT-23, CIC-IDS, Bot-IoT)
- Research paper authors
- Open source community contributions

## 📞 Contact

For questions and support, please open an issue in the GitHub repository. 