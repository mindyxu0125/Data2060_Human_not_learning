# Data2060_Human_not_learning

## CART Classification Model -- From Scratch

This repository contains an from-scratch implementation of the CART (Classification and Regression Trees) algorithm for classification.
The project was developed as the final assignment for Data 2060 at Brown University.

Our implementation includes:

- A custom DecisionTreeCART classifier implemented entirely from scratch

- Node-level data structures storing splits, class distributions, and predictions

- Gini and entropy impurity functions matching scikit-learn’s formulas

- Full CART training procedure:

    - Best-split search across all features

    - Threshold scanning using sorted feature values

    - Weighted-impurity evaluation

    - Recursive tree building with stopping rules

- Support for binary and multi-class classification

- Prediction by tree traversal, returning majority labels

- Accuracy evaluation utilities

- Internal tests verifying correctness of impurity and prediction logic

- An experimental section demonstrating model performance on a real dataset

## Python Version and Package Versions
This project was developed and tested using the following environment.
Reproducing these exact versions is recommended to ensure consistency with our results:


| Dependency       | Version |
|------------------|---------|
| **Python**       | 3.13.5  |
| **NumPy**        | 2.1.3   |
| **Matplotlib**   | 3.10.0  |
| **scikit-learn** | 1.6.1   |
| **pytest**       | 8.3.4   |

These versions are captured in the accompanying environment.yaml file so that the entire environment can be recreated using:
```bash
conda env create -f environment.yaml
conda activate cart-env
```

## Project Directory Structure
This repository follows a clean, modular structure to separate data, implementation code, and documentation.  
Below is the directory layout for the CART classification project:
├── data/ # Datasets used for training, testing, and evaluation

├── src/ # Source code implementation of the CART classifier

├── .gitignore # Git ignore rules for temporary and system files

├── LICENSE # Open-source license (MIT License)

├── README.md # Project documentation and usage guide

├── presentation.pdf # Final presentation slides for the project

└── report.pdf # Final written report describing methods and results


