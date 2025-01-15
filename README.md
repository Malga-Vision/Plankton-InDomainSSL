# In-domain self-supervised learning for plankton image classification on a budget
Official PyTorch implementation of the paper 

# Abstract


# Getting Started
## Installation
### Install Python Packages

To install the required Python packages, run the following command:

```bash
pip install -r requirements.txt
```

## Running Pipeline
To run the pipeline, follow these steps:

1. Make the `extract_features.sh` script executable if it is not using the following command:
    ```bash
    chmod +x extract_features.sh
    ```
2. Check the paths in `extract_features.sh` and `train_classifier.sh` scripts to match the location on your local machine. Otherwise, change them to the correct location of the dataset and model checkpoint.
3. Run the `extract_features.sh` script:
    ```bash
    ./extract_features.sh
    ```
    The script runs the feature extraction, classifier training and test classification 3 times. Parameters can be changed as well in the bash scripts.

# Download Pre-Trained models

All the pre-trained models used in the study are available at the following link:

[Pre-trained models checkpoint folder](https://drive.google.com/file/d/1SwUZ8rDJqn3aMSpCmB4uc9G9d1ba-GEm/view?usp=sharing)


# Citation

Accepted for publication in the proceedings of the 3rd Workshop on Maritime Computer Vision (MaCVi 2025).

### ArXiv Bibtex
