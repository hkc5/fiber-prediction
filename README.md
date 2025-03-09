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
├── notebooks/      # Exploratory Jupyter notebooks
│   ├── HOG/       # HOG feature analysis notebooks
│   └── NN/        # Neural network analysis notebooks
│
├── models/        # Trained models
│   ├── cnn/       # CNN model weights
│   └── svr_hog/   # SVR-HOG model weights
│
├── images/        # Image data
├── environment.yml # Conda environment configuration
└── README.md      # Project overview
```

## Environment Setup

This project uses Conda for environment management. The environment includes PyTorch for deep learning, scikit-learn for machine learning, OpenCV for image processing, and other scientific computing libraries.

### Prerequisites
- Conda (Miniconda or Anaconda)

### Creating the Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/fiber-orientation.git
cd fiber-orientation

# Create and activate conda environment
conda env create -f environment.yml
conda activate fiber-orientation
```

### Key Dependencies
- Python 3.12
- PyTorch & TorchVision
- scikit-learn & scikit-image
- OpenCV
- NumPy & Pandas
- Jupyter Notebook

## Usage

### Jupyter Notebooks
The project includes several Jupyter notebooks for analysis:

- `notebooks/CNN_Inference.ipynb`: CNN model inference and visualization
- `notebooks/HOG_Inference.ipynb`: HOG-based model inference
- `notebooks/EDA.ipynb`: Exploratory Data Analysis

The `notebooks/HOG/` and `notebooks/NN/` directories contain detailed analysis notebooks for each approach.

### Models
Pre-trained models are available in the `models/` directory:
- CNN models in `models/cnn/`
- SVR-HOG models in `models/svr_hog/`

Each model type includes various versions (raw, augmented, balanced) for different use cases.

## License
MIT License - see [LICENSE](LICENSE) file for details.
