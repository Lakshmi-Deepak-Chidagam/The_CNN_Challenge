# ITCS 6169/8169 — Assignment 1: The CNN Challenge

## Overview

This repository contains my solution for Assignment 1 of ITCS 6169/8169
Computer Vision.

The objective is to develop and experimentally evaluate CNN-based approaches
for a 16-class scene-classification problem using a limited training dataset.

The project follows a hypothesis-driven experimental process. Model and
hyperparameter decisions are based on validation performance rather than
repeated evaluation on the held-out test set.

## Final Result

The selected model is an **ImageNet-pretrained ConvNeXt-Tiny**.

- **Best validation accuracy: 97.08%**
- **Final test accuracy: 93.25%**
- **Test correct: 373 / 400**
- **Final test loss: 0.2263**
- Best validation epoch: **2**

The final model was selected using validation performance before the held-out
test evaluation was performed.


## Dataset

The provided dataset contains:

- 2,400 training images
- 400 test images
- 16 scene categories

The provided training set was divided using an 80/20 split:

- Training: 1,920 images
- Validation: 480 images
- Random seed: 0

The same train/validation split was preserved across controlled experiments
where applicable.


## Experimental Journey

| Experiment | Model | Main Change | Best Validation Accuracy | Best Epoch |
|---|---|---|---:|---:|
| 0 | Starter TNet | Original baseline | 49.79% | 11 |
| 1 | ResNet18 | ImageNet transfer learning + RGB 224×224 | 93.33% | 2 |
| 2 | ResNet18 | RandomResizedCrop + RandomHorizontalFlip | 93.12% | 4 |
| 3 | ResNet18 | Cosine learning-rate scheduling | 93.13% | 5 |
| 4 | ResNet50 | Increased CNN capacity | 94.37% | 7 |
| **5** | **ConvNeXt-Tiny** | **Changed pretrained CNN architecture** | **97.08%** | **2** |
| 6 | ConvNeXt-Tiny | Label smoothing = 0.1 | 96.88% | 3 |
| 7 | ConvNeXt-Tiny | Increased input resolution to 320×320 | 96.46% | 2 |
| 8 | ConvNeXt-Tiny | Horizontal flip + mild ColorJitter | 96.88% | 2 |

Detailed hypotheses, configurations, observations, and conclusions are
available in `EXPERIMENTS.md`.


## Experimental Findings

### Starter CNN

The starter CNN achieved only **49.79% validation accuracy**. Training loss
continued decreasing while validation performance stopped improving,
indicating substantial overfitting.

### Transfer Learning

Switching to an ImageNet-pretrained ResNet18 with 224×224 RGB input increased
validation accuracy to **93.33%**.

This was the largest single improvement observed during the experimental
sequence and demonstrated the effectiveness of the overall transfer-learning
pipeline for the limited dataset.

### Data Augmentation

RandomResizedCrop and RandomHorizontalFlip produced **93.12%** validation
accuracy with ResNet18 and did not improve upon the non-augmented
configuration.

### Learning-Rate Scheduling

CosineAnnealingLR produced **93.13%** validation accuracy and did not improve
upon the fixed learning-rate ResNet18 experiment.

### Increased CNN Capacity

ImageNet-pretrained ResNet50 increased validation accuracy to **94.37%**.
However, training loss approached zero while validation loss later increased,
showing that overfitting remained.

### ConvNeXt-Tiny

Changing the CNN architecture to ImageNet-pretrained ConvNeXt-Tiny produced
the strongest result: **97.08% validation accuracy**.

The best validation result occurred at epoch 2. Additional training reduced
training loss but did not improve validation accuracy.

### Label Smoothing

Adding label smoothing of 0.1 produced **96.88% validation accuracy**.
Although it reduced model confidence and prevented training loss from
approaching zero, it did not improve validation accuracy.

### Higher Resolution

Increasing input resolution from 224×224 to 320×320 produced **96.46%**
validation accuracy while requiring substantially more computation.

### Mild Augmentation

Adding RandomHorizontalFlip and mild ColorJitter to ConvNeXt-Tiny produced
**96.88% validation accuracy**, slightly below the non-augmented model.

These results led to retaining the simpler Experiment 5 configuration.


## Final Model Configuration

- Architecture: **ConvNeXt-Tiny**
- Pretraining: **ImageNet**
- Number of output classes: **16**
- Input resolution: **224 × 224**
- Input representation: **RGB**
- Normalization: **ImageNet mean and standard deviation**
- Training augmentation: **None**
- Optimizer: **Adam**
- Learning rate: **0.0001**
- Loss: **CrossEntropyLoss**
- Epochs: **10**
- Best epoch: **2**
- Random seed: **0**
- Best validation accuracy: **97.08%**
- Final test accuracy: **93.25%**
- Final test loss: **0.2263**


## Validation Error Analysis

The selected model correctly classified **466 of 480 validation images**.

The largest remaining source of error was the Bedroom class. Five Bedroom
images were predicted as LivingRoom and one as Kitchen.

Other errors primarily involved visually related scene categories such as
Mountain and OpenCountry.

Nine of the sixteen classes achieved 100% validation accuracy.

This suggests that the remaining errors primarily involve fine-grained
distinctions between visually similar scenes.


## Final Test Evaluation

After Experiment 5 was selected using validation performance, the checkpoint
was evaluated on the held-out test set.

The test preprocessing uses:

- RGB input
- Resize to 224 × 224
- ImageNet normalization
- No random augmentation

Results:

- Test images: **400**
- Correct predictions: **373**
- Incorrect predictions: **27**
- Test accuracy: **93.25%**
- Test loss: **0.2263**

The test result was not used for further model or hyperparameter selection.


## Repository Structure

```text
.
├── README.md
├── AI_USAGE.md
├── EXPERIMENTS.md
├── requirements.txt
├── notebooks/
│   └── ITCS_6169_8169_Assignment1.ipynb
├── src/
│   ├── train.py
│   └── evaluate.py
├── checkpoints/
│   └── README.md
└── report/
    └── Assignment1_Report.pdf
```


## Environment

Experiments were conducted using Google Colab with GPU acceleration.

Observed environment:

- Python
- PyTorch
- torchvision
- NumPy
- Matplotlib
- NVIDIA T4 GPU

Project dependencies are listed in `requirements.txt`.


## Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```


## Expected Dataset Structure

The training and test data should follow a PyTorch `ImageFolder`-compatible
directory structure:

```text
data/
├── train/
│   ├── Bedroom/
│   ├── Coast/
│   ├── Flower/
│   └── ...
└── test/
    ├── Bedroom/
    ├── Coast/
    ├── Flower/
    └── ...
```

The 16 classes are:

```text
Bedroom
Coast
Flower
Forest
Highway
Industrial
InsideCity
Kitchen
LivingRoom
Mountain
Office
OpenCountry
Store
Street
Suburb
TallBuilding
```


## Reproduction

### Train the Final Configuration

```bash
python src/train.py --data-root <path_to_training_data>
```

The training script reproduces the selected ConvNeXt-Tiny configuration and
saves the checkpoint corresponding to the best validation accuracy.

### Evaluate a Checkpoint

```bash
python src/evaluate.py \
    --data-root <path_to_test_data> \
    --checkpoint <path_to_checkpoint>
```

Evaluation uses deterministic 224×224 RGB preprocessing with ImageNet
normalization.


## Final Checkpoint

The selected checkpoint is:

```text
convnext_tiny_exp5_best.pth
```

The checkpoint corresponds to the best-validation state from Experiment 5:

- ConvNeXt-Tiny
- ImageNet pretrained initialization
- 16 output classes
- 224×224 RGB input
- Best validation accuracy: 97.08%
- Best epoch: 2

Because the checkpoint is approximately **106 MB**, it may be hosted
externally rather than committed directly to the Git repository.

See `checkpoints/README.md` for checkpoint access information.


## Reproducibility

The experiments use random seed:

```text
0
```

The same train/validation split is preserved for controlled comparisons where
applicable.

Model-selection decisions are based on validation performance rather than
repeated evaluation on the test set.

The final checkpoint stores metadata describing the selected architecture,
class names, input size, validation accuracy, best epoch, and random seed.


## AI-Assisted Development

ChatGPT (OpenAI) was used as an AI-assisted learning, coding, debugging, and
documentation tool.

AI-generated suggestions were not assumed to be correct. Implementations and
experimental recommendations were checked using tensor dimensions, active
training configurations, learning curves, validation results, and runtime
behavior.

One important verification occurred during the label-smoothing experiment,
where an initially ineffective criterion implementation was identified,
corrected, and rerun before drawing a conclusion.

A detailed record of AI usage, verification, an ineffective AI-assisted
implementation, and human experimental decisions is provided in:

```text
AI_USAGE.md
```


## Experiment Documentation

Detailed hypotheses, configurations, results, failure analysis, validation
error analysis, and experimental decisions are documented in:

```text
EXPERIMENTS.md
```


## Final Summary

The experimental process improved best validation accuracy from **49.79%**
with the starter CNN to **97.08%** with ImageNet-pretrained ConvNeXt-Tiny.

The final selected model achieved **93.25% accuracy on the held-out test set**.

The experiments showed that transfer learning and CNN architecture choice
provided the largest improvements, while additional label smoothing, higher
input resolution, and mild augmentation did not improve upon the selected
ConvNeXt-Tiny configuration.
