
# Land Cover Classification using CNN on EuroSAT Dataset

## Project Overview
This project applies a Convolutional Neural Network (CNN) model based on **ResNet18** to classify satellite images from the **EuroSAT** dataset into different land cover categories.

- **Problem**: Land Cover Classification from satellite imagery.
- **Dataset**: [EuroSAT](https://github.com/phelber/eurosat) (RGB version).
- **Model**: Pre-trained ResNet18 fine-tuned on EuroSAT.
- **Framework**: PyTorch.

## Directory Structure
```
/Final_Project_Submission/
    ├── eurosat_cnn.py
    ├── resnet18_eurosat.pth
    ├── EuroSAT/ (optional if dataset download is automatic)
    └── README.md
```

## Setup Instructions

1. Install the required libraries:
```bash
pip install torch torchvision scikit-learn seaborn matplotlib
```

2. Run the Python script:
```bash
python eurosat_cnn.py
```

The script will:
- Download or load the EuroSAT dataset.
- Train the ResNet18 model for 10 epochs.
- Evaluate the model on a test set.
- Display training and validation accuracy plots.
- Plot a confusion matrix.
- Save the trained model (`resnet18_eurosat.pth`).

## Requirements
- Python 3.8+
- PyTorch
- Torchvision
- Scikit-learn
- Seaborn
- Matplotlib

(Optional: CUDA-enabled GPU for faster training.)

## Results
The fine-tuned model achieved an excellent **Test Accuracy of approximately 94%** on the EuroSAT dataset.
- **Train Accuracy**: ~93.54%
- **Validation Accuracy**: ~94.69%
- **Classification Report**:
  - Precision and Recall above 90% for major classes like Forest, SeaLake, Residential.

Confusion matrix and accuracy curves demonstrate strong model performance across different land cover types.

## Contributors
- Satyawan Singh
- Shreyash Wankhade

## Acknowledgements
- EuroSAT Dataset by Patrick Helber.
- PyTorch and Torchvision libraries.
