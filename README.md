# Lumen — Natural Scene Classification with CNN

A Convolutional Neural Network built entirely from scratch to classify natural scene images into six categories — buildings, forest, glacier, mountain, sea, and street — achieving **90.10% test accuracy** on unseen data.

## Overview
Lumen (Latin for "light") explores how deep learning models can learn to truly "see" and understand visual scenes — not just classify pixels, but recognize the patterns, textures, and structures that define a place. Using the Intel Image Classification Dataset (~25,000 images), a CNN architecture was designed, trained, and tuned entirely from scratch, without relying on pre-trained networks.

## Dataset
- **Source:** [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (Kaggle)
- **Size:** ~25,000 RGB images, 150×150 pixels
- **Classes:** buildings, forest, glacier, mountain, sea, street
- **Split:** 11,932 training / 2,102 validation / 3,000 test images

| Class | Train | Test |
|---|---|---|
| buildings | 1,863 | 437 |
| forest | 1,931 | 474 |
| glacier | 2,044 | 553 |
| mountain | 2,136 | 525 |
| sea | 1,933 | 510 |
| street | 2,025 | 501 |

## Model Architecture
A custom CNN built from scratch with 4 convolutional blocks (no pre-trained models used):

- **Input:** 128×128×3 RGB images
- **Block 1–4:** Conv2D layers (32 → 256 filters) with Batch Normalization and Max Pooling
- **Global Average Pooling** (instead of Flatten) to reduce parameters
- **Dense layer** (256, ReLU) with Batch Normalization and Dropout
- **Output:** Dense(6, Softmax) for the six scene classes
- **Total parameters:** 653,350 (651,430 trainable)

## Preprocessing
- Images resized from 150×150 to 128×128
- Pixel normalization ([0,255] → [0,1])
- Data augmentation on training data only (rotation, shift, zoom, shear, flip, brightness)
- One-hot label encoding, 15% validation split, batch loading (batch size 32)

## Training & Experimentation
Two tuning experiments were run to compare hyperparameter configurations:

| Experiment | Learning Rate | Dropout | Max Epochs | Test Accuracy |
|---|---|---|---|---|
| **1 (Best)** | 1e-3 | 0.4 | 45 | **90.10%** |
| 2 | 5e-4 | 0.5 | 30 | 88.53% |

- **Loss:** Categorical cross-entropy with label smoothing (0.1)
- **Optimizer:** Adam
- **Regularization:** Dropout, Batch Normalization, Data Augmentation, EarlyStopping, ReduceLROnPlateau

## Results
**Final Test Accuracy: 90.10%** on 3,000 unseen images

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| buildings | 0.88 | 0.91 | 0.90 |
| forest | 0.98 | 0.99 | 0.98 |
| glacier | 0.85 | 0.86 | 0.85 |
| mountain | 0.87 | 0.83 | 0.85 |
| sea | 0.92 | 0.93 | 0.92 |
| street | 0.91 | 0.90 | 0.91 |

Forest was classified most reliably (F1: 0.98), while glacier and mountain were the most commonly confused pair due to their visual similarity — a pattern also reflected in the confusion matrix.

## Challenges & Future Work
- Glacier vs. mountain and street vs. buildings were the most visually ambiguous pairs to classify
- Slight class imbalance across categories may have marginally affected results
- Future improvement: additional targeted data augmentation for the most confused classes, without relying on pre-trained models

## Tech Stack
- **Language:** Python
- **Framework:** TensorFlow / Keras
- **Environment:** Google Colab (GPU-accelerated training)
- **Tools:** NumPy, Matplotlib, scikit-learn (evaluation metrics)

## Author
Leen Alsahli — [LinkedIn](https://linkedin.com/in/leen-alsahli-1064a6305) | [Portfolio](https://leen-portfolio-inky.vercel.app)
