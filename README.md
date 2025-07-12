# Automated Brain Tumor Detection and Classification from MRI Scans using Deep Learning

This project presents a deep learning-based system for the automated detection and classification of brain tumors from MRI images. It was developed as part of our Bachelor of Science in Information Technology (BSIT) final year project at GSCWU, Bahawalpur (Session 2021–2025).

## Problem Statement

Manual diagnosis of brain tumors from MRI scans is time-consuming and error-prone. Accurate early detection is critical for effective treatment planning. This project addresses the need for an automated, reliable classification system to assist radiologists in identifying four classes:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

##  Project Overview

- **Dataset**: Publicly available T1-weighted contrast-enhanced brain MRI images, categorized into four classes.
- **Models Used**:
  - **Custom CNN**: A 5-layer Convolutional Neural Network with dropout regularization.
  - **Transfer Learning with ResNet18**: Utilized pre-trained ImageNet weights, followed by fine-tuning on domain-specific data.

## Tech Stack

- **Languages & Tools**: Python, Jupyter Notebook, PyTorch, Matplotlib, Scikit-learn
- **Web Deployment**: Flask (for the backend interface)
- **Libraries Used**:
  - `torch`, `torchvision`
  - `matplotlib`, `seaborn`
  - `sklearn`
  - `pandas`, `numpy`

## Methodology

- **Preprocessing**:
  - Resizing images to 224×224
  - RGB conversion (for ResNet compatibility)
  - Normalization using ImageNet standards
- **Data Augmentation**:
  - Random flips, affine transformations, brightness adjustments
- **Training**:
  - Custom CNN trained from scratch
  - ResNet18 fine-tuned by unfreezing selected layers
- **Evaluation**:
  - Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## Results

The fine-tuned ResNet18 outperformed all models with:
- **Accuracy**: 99.54%
- **F1-Score**: 1.00 (macro & weighted average)
- **Precision & Recall**: Near-perfect across all classes

Performance comparisons were visualized through:
- Training/Validation curves
- Confusion matrices
- Sample predictions

## Visuals

Sample screenshots and confusion matrices can be found in the notebook.

## Project Structure
```
BrainTumorDetectionSystem/
├── brain_tumor_app/ # Flask app backend
│ ├── app.py
│ ├── templates/
│ └── static/
│ └── brain_tumor_model_full.pth # model file
│
├── notebook/ # Model training and evaluation
│ └── BrainTumorDetectionSystemFYP25.ipynb
│
├── requirements.txt # Python dependencies
├── README.md
```
## Authors

This project was developed as part of the Final Year Project (Session 2021–2025) at  
**The Government Sadiq College Women University, Bahawalpur**.

- **Amna Ghaffar**
  *Group Leader — Led the development of the entire system, including model training, Flask backend, and web interface.*

- **Alishba Gul**
  *Contributed to project documentation and report writing.*

- **Iqra Jameel**
  *Participated in discussions and presentation sessions.*
