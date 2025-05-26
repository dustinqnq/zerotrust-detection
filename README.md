# Multi-Stage Zero-Trust IoT Detection System

## 🌟 Overview

This project implements a sophisticated multi-stage zero-trust IoT detection system that combines multiple machine learning algorithms to detect and classify IoT-based threats, with a particular focus on Mirai botnet variants. The system achieves high accuracy (99.939%) and low false positive rates (0.021%) through its innovative four-stage detection architecture.

## 🎯 Key Features

- **Four-Stage Detection Architecture**
  - Boundary Detection (Isolation Forest)
  - Behavior Analysis (Random Forest)
  - Anomaly Detection (One-Class SVM)
  - Threat Intelligence Analysis (Neural Network)

- **Advanced Feature Engineering**
  - Intelligent feature selection from 80 to 37 optimal features
  - Multiple selection algorithms (F-score, Mutual Information, Random Forest)
  - Automated feature importance analysis

- **Real-time Threat Intelligence**
  - Dynamic threat intelligence updates
  - Mirai variant detection
  - Attack pattern recognition
  - Vulnerability tracking

## 📊 Performance Metrics

- Classification Accuracy: 99.939%
- Threat Detection Rate: 99.609%
- False Positive Rate: 0.021%
- Processing Speed: 550,533 samples/second

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Virtual Environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/iotdetection.git
cd iotdetection
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

1. Download the CIC-IoT-2023 dataset
2. Place the dataset files in the `data/cic_iot_2023/` directory

### Running the System

1. Start the main detection engine:
```bash
python src/enhanced_multi_stage_detector.py
```

2. Update threat intelligence:
```bash
python threat_intelligence/threat_intelligence_updater.py
```

3. View performance metrics:
```bash
python utils/performance_evaluation.py
```

## 📁 Project Structure

```
iotdetection/
├── src/                                    # Core source code
│   ├── enhanced_multi_stage_detector.py    # Main detection engine
│   ├── advanced_feature_optimizer.py       # Feature optimization
│   └── cic_iot_data_processor.py          # Data processing
├── threat_intelligence/                    # Threat intelligence module
├── utils/                                 # Utility functions
├── data/                                  # Dataset directory
├── models/                                # Model storage
├── results/                               # Results and metrics
└── docs/                                  # Documentation
```

## 🔧 Configuration

The system can be configured through various JSON files:

- `threat_intelligence/threat_intel_config.json`: Threat intelligence settings
- `models/config/model_config.json`: Model parameters
- `data/config/preprocessing_config.json`: Data preprocessing settings

## 📈 Results and Visualization

Results and performance metrics are stored in the `results/` directory:

- Performance charts
- Confusion matrices
- ROC curves
- Detailed performance reports

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CIC-IoT-2023 dataset providers
- Contributors and researchers in the IoT security field
- Open source community

## 📞 Contact

For questions and support, please open an issue in the GitHub repository. 