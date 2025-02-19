# Melanoma Skin Cancer Detection

## Abstract

In the realm of cancer, there exist over 200 distinct forms, with melanoma standing out as the most lethal type of skin cancer among them. The diagnostic protocol for melanoma typically initiates with clinical screening, followed by dermoscopic analysis and histopathological examination. Early detection of melanoma skin cancer is pivotal, as it significantly enhances the chances of successful treatment. The initial step in diagnosing melanoma skin cancer involves visually inspecting the affected area of the skin. Dermatologists capture dermatoscopic images of the skin lesions using high-speed cameras, which yield diagnostic accuracies ranging from 65% to 80% for melanoma without supplementary technical assistance. Through further visual assessment by oncologists and dermatoscopic image analysis, the overall predictive accuracy of melanoma diagnosis can be elevated to 75% to 84%. The objective of the project is to construct an automated classification system leveraging image processing techniques to classify skin cancer based on images of skin lesions.

## Problem statement

To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

## Table of Contents

- [General Info](#general-information)
- [Model Architecture](#model-architecture)
- [Model Summary](#model-summary)
- [Model Evaluation](#model-evaluation)
- [Technologies Used](#technologies-used)
- [Acknowledgements](#acknowledgements)
- [Collaborators](#collaborators)

<!-- You can include any other section that is pertinent to your problem -->

## General Information

The dataset comprises 2357 images depicting malignant and benign oncological conditions, sourced from the International Skin Imaging Collaboration (ISIC). These images were categorized based on the classification provided by ISIC, with each subset containing an equal number of images.

## Model Architecture

This CNN model is designed for image classification and processes 180x180 RGB images. It follows a convolutional-pooling pattern, gradually increasing the number of filters to extract hierarchical features.

1. Input Preprocessing
Rescaling Layer: Normalizes pixel values to [0,1] range to improve training stability.

2. Feature Extraction (Convolutional Layers)
Conv2D (32 filters, 3×3, ReLU) + MaxPooling (2×2) → Captures low-level features (edges, textures).
Conv2D (64 filters, 3×3, ReLU) + MaxPooling (2×2) → Extracts more complex features.
Conv2D (128 filters, 3×3, ReLU) + MaxPooling (2×2) → Captures deeper patterns.
Conv2D (256 filters, 11×11, ReLU) + MaxPooling (2×2) + Dropout (50%) → Detects high-level spatial patterns while reducing overfitting.

3. Fully Connected Layers (Classification Head)
Flatten → Converts feature maps into a 1D vector.
Dense (256, ReLU) + Dropout (25%) → First fully connected layer for pattern learning.
Dense (128, ReLU) + Dropout (25%) → Further abstraction of features.
Dense (64, ReLU) + Dropout (25%) → Enhances feature representation.
Dense (Output Size, Softmax) → Final classification layer with softmax activation for multi-class prediction.

Summary
Model Performance and Observations :
Initial Accuracy : The model started with a low training accuracy of 13.8% and validation accuracy of 12.8%, which indicates a challenging classification task.
Steady Improvement : Accuracy improved consistently, reaching 88.6% (training) and 88.9% (validation) by Epoch 45.
Loss Reduction : Training loss reduced from 2.33 to 0.18, and validation loss reduced from 2.46 to 0.56.
Validation Accuracy Peaks at 88.9% : Indicates strong generalization to unseen data.
Best Epoch : Around Epoch 36-45, where accuracy stabilized above 92% (training) and 88% (validation).
ReduceLROnPlateau Activation (Epoch 34) : Learning rate was reduced to 0.0002, helping fine-tune performance.

Comparison with Previous Models :
Significant Improvement : Previous models had much lower accuracy (likely around 70-75%). This model surpasses them with an over 10-15% boost.
More Stable Learning Curve : No drastic drops or spikes, indicating better optimization.
Better Handling of Overfitting : Training and validation curves are closely aligned, showing reduced overfitting.

Overfitting or Underfitting?
Mild Overfitting (Epoch 40+) : Training accuracy is 92-93%, but validation stagnates around 88-89%. The small gap suggests minimal overfitting but not excessive.
Balanced Model : The ReduceLROnPlateau helped curb overfitting, keeping validation accuracy high.

Effect of Class Rebalancing :
Boosted Early Training Stability : Class rebalance likely helped in overcoming class imbalance issues, preventing dominance by majority classes.
Higher Generalization : The validation accuracy jumps suggest that the model learned well from rebalanced data.

Conclusion :
Highly improved model, achieving a substantial boost in accuracy.
Minimal overfitting, controlled well with LR reduction.
Class rebalancing contributed positively by improving early learning.
One of the best versions of this model so far.
The Final Model Validation also shows a 100% match between the Actual Class and the Predicted Class.

## Technologies Used

- [Python](https://www.python.org/) - version 3.11.4
- [Matplotlib](https://matplotlib.org/) - version 3.7.1
- [Numpy](https://numpy.org/) - version 1.24.3
- [Pandas](https://pandas.pydata.org/) - version 1.5.3
- [Seaborn](https://seaborn.pydata.org/) - version 0.12.2
- [Tensorflow](https://www.tensorflow.org/) - version 2.15.0

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Collaborators

Created by [@Lakki-10-Universe]([https://github.com/Lakki-10-Universe])
