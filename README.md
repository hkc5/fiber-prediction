# Fiber Orientation Prediction Project

## Overview
Advanced machine learning project for predicting and analyzing fiber orientations using state-of-the-art image processing and neural network techniques.

## Project Structure
```
fiber-orientation/
│
├── src/
│   └── fiber_predictor/
│       ├── apps/           # Application-specific modules
│       ├── neural_networks/# Neural network models
│       ├── svr_hog/        # Support Vector Regression models
│       └── utils/          # Utility functions
│
├── tests/          # Unit tests
├── notebooks/      # Exploratory Jupyter notebooks
├── data/           # Data storage
│   ├── raw/        # Original data
│   ├── processed/  # Preprocessed data
│   └── models/     # Trained models
│
├── scripts/        # Utility and training scripts
├── docs/           # Project documentation
│
├── LICENSE         # Project license
└── README.md       # Project overview
```

## Prerequisites
- Python 3.8+
- Conda (recommended)

## Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/fiber-orientation.git
cd fiber-orientation

# Create and activate conda environment
conda env create -f environment.yml
conda activate fiber-orientation
```

## Usage

### Prediction
```python
from src.fiber_predictor.utils.utils import get_feature_vector
import cv2

# Load and process an image
image = cv2.imread('path/to/image.png')
features = get_feature_vector(image)
```

### Training
```bash
# Run training script
python scripts/train_model.py \
    --data-dir data/raw \
    --csv-dir data/processed
```

## Documentation
- [Installation Guide](docs/installation.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## License
MIT License - see [LICENSE](LICENSE) file for details.
