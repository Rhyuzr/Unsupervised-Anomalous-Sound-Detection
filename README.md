# Unsupervised-Anomalous-Sound-Detection


An unsupervised machine learning solution for detecting anomalous sounds in industrial environments, specifically targeting slide rail machines.

## Overview

This project implements an autoencoder-based approach for detecting anomalous sounds in industrial settings. The system is trained exclusively on normal sound samples, enabling it to identify deviations from the expected behavior without explicit anomaly labels.

## Features

- Unsupervised learning using autoencoder architecture
- Industrial machine sound analysis (slide rail)
- Anomaly detection through reconstruction error
- Audio preprocessing and feature extraction
- Performance evaluation metrics

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Librosa
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- tqdm

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install tensorflow librosa numpy pandas matplotlib seaborn scikit-learn tqdm
```

## Project Structure

```
.
├── data/              # Audio dataset directory
│   ├── normal/       # Normal sound samples
│   └── anomaly/      # Anomalous sound samples (optional for testing)
├── AML2_Unsupervised_Anomalous_Sound_Detection.py  # Main script
└── README.md
```

## Usage

1. Place your audio files in the appropriate directories:
   - Normal sound samples: `data/normal/`
   - Anomalous sound samples (optional): `data/anomaly/`

2. Run the main script:
```bash
python AML2_Unsupervised_Anomalous_Sound_Detection.py
```

## Technical Details

### Architecture
- Autoencoder-based neural network
- Convolutional layers for feature extraction
- Reconstruction loss for anomaly detection
- Early stopping and model checkpointing

### Metrics
- Reconstruction error
- ROC AUC score
- ROC curve visualization

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request


## Acknowledgments

- This project is based on the DCASE Challenge for anomalous sound detection
- Special thanks to the Librosa team for their excellent audio processing library
