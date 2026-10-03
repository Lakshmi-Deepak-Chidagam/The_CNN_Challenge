# ITCS 6169/8169 Assignment 1 — Experiment Log

## Experimental Protocol

The goal of this project is to improve image classification performance on a
16-class scene classification dataset while maintaining a controlled and
reproducible experimental process.

The provided training set contains 2,400 images. I use an 80/20 split,
resulting in:

- Training samples: 1,920
- Validation samples: 480
- Test samples: 400
- Number of classes: 16
- Random seed: 0

The validation set is used for model and hyperparameter decisions. The test
set is reserved for final evaluation after the final model has been selected.

Experiments are designed around specific hypotheses rather than changing many
components without analysis.


# Experiment 0 — Starter CNN Baseline

## Objective

Establish the performance of the provided starter CNN before introducing any
changes.

## Configuration

- Architecture: Starter TNet CNN
- Input resolution: 64 × 64
- Input channels: Grayscale
- Batch size: 64
- Optimizer: Adam
- Learning rate: 0.002
- Epochs: 20
- Loss: CrossEntropyLoss
- Random seed: 0
- Trainable parameters: 57,776
- Training samples: 1,920
- Validation samples: 480
- Device: Google Colab T4 GPU

## Results

- Best validation accuracy: **49.79%**
- Best epoch: **11**
- Final epoch validation accuracy: **46.67%**
- Initial training loss: **2.5592**
- Final training loss: **0.1181**
- Final validation loss: **2.2756**
- Baseline test accuracy produced by starter notebook: **42.25%**
- Test loss: **1.8582**
- Approximate training time: **60.4 seconds**

## Observation

Training loss decreased consistently throughout training, from 2.5592 to
0.1181. However, validation loss initially decreased and later increased.
Validation accuracy reached its highest value of 49.79% at epoch 11 and did
not improve consistently afterward.

This behavior indicates that the model was increasingly fitting the training
data without obtaining a corresponding improvement on unseen validation
samples.

## Interpretation

The baseline demonstrates substantial overfitting and limited generalization.

The provided CNN contains only one convolutional layer and processes images
as 64 × 64 grayscale inputs. This representation may discard useful scene
information such as color, texture, and higher-level spatial patterns.

## Decision

The next experiment investigates transfer learning using an
ImageNet-pretrained ResNet18. RGB input and pretrained convolutional features
are used to determine whether a richer visual representation improves
generalization on the relatively small training dataset.


# Experiment 1 — ImageNet-Pretrained ResNet18

## Hypothesis

Experiment 0 showed substantial overfitting and limited generalization. The
starter CNN learned visual features entirely from only 1,920 training images
and used low-resolution grayscale input.

I studied transfer learning and ImageNet-pretrained convolutional neural
networks to understand how previously learned visual features can help when
the available labeled dataset is small. I hypothesized that a pretrained
ResNet18 using higher-resolution RGB images would provide a substantially
stronger representation than the shallow CNN trained from scratch.

## Controlled Changes

Compared with Experiment 0:

- Architecture changed from the starter TNet to ResNet18.
- ImageNet-pretrained weights were used.
- Input representation changed from grayscale to RGB.
- Input resolution changed from 64 × 64 to 224 × 224.
- ImageNet normalization was used.
- Learning rate changed from 0.002 to 0.0001.
- Training duration was changed from 20 to 10 epochs.

The dataset, 80/20 train-validation ratio, random seed, and classification
objective were preserved. The same train/validation indices were reconstructed
using random seed 0.

Because several components changed together, this experiment evaluates the
combined transfer-learning configuration rather than attributing the
improvement to a single individual change.

## Configuration

- Architecture: ResNet18
- Pretraining: ImageNet
- Input resolution: 224 × 224
- Input channels: RGB
- Normalization: ImageNet mean and standard deviation
- Training samples: 1,920
- Validation samples: 480
- Batch size: 64
- Optimizer: Adam
- Learning rate: 0.0001
- Loss: CrossEntropyLoss
- Epochs: 10
- Random seed: 0
- Device: Google Colab NVIDIA T4 GPU

## Results

- Best validation accuracy: **93.33%**
- Best epoch: **2**
- Final validation accuracy: **92.92%**
- Final training loss: **0.0059**
- Final validation loss: **0.2201**
- Approximate training time: **94.3 seconds**
- Absolute improvement over Experiment 0: **43.54 percentage points**

## Observation

The pretrained ResNet18 produced a dramatic improvement over the starter
baseline. Validation accuracy reached 91.46% after the first epoch and first
reached its maximum of 93.33% at epoch 2.

Training loss continued decreasing throughout training and reached 0.0059 at
epoch 10. Validation accuracy, however, remained approximately 92–93% after
the initial improvement. Validation loss stabilized near 0.22 rather than
following training loss toward zero.

## Interpretation

The experiment demonstrates that the combined transfer-learning configuration
generalizes substantially better than the starter CNN. The pretrained model,
deeper CNN architecture, RGB information, and higher input resolution
collectively provide a much stronger representation for the scene
classification task.

The near-zero training loss combined with a relatively stable validation
accuracy also indicates that some overfitting remains. This motivated testing
training-time data augmentation as a regularization strategy.

## Decision

ResNet18 was retained as the current strongest architecture. Experiment 2
tests whether adding moderate training-time data augmentation can improve
generalization while keeping the ResNet18 transfer-learning configuration.


# Experiment 2 — ResNet18 with Data Augmentation

## Hypothesis

Experiment 1 achieved 93.33% validation accuracy, but training loss decreased
to nearly zero while validation accuracy plateaued around 92–93%.

I studied data augmentation as a regularization technique and investigated
whether increasing training-image diversity could improve generalization.

The hypothesis was that moderate random cropping and horizontal flipping would
make the model less dependent on the exact training images and potentially
increase validation accuracy.

## Controlled Changes

Experiment 2 retained:

- ImageNet-pretrained ResNet18
- 224 × 224 RGB input
- ImageNet normalization
- Adam optimizer
- Learning rate 0.0001
- CrossEntropyLoss
- Batch size 64
- 10 epochs
- Random seed 0
- Exact training and validation indices from Experiment 1

The primary change was the addition of training-time augmentation:

- RandomResizedCrop(224, scale=(0.8, 1.0))
- RandomHorizontalFlip(p=0.5)

Validation images were not randomly augmented.

## Results

- Best validation accuracy: **93.12%**
- Best epoch: **4**
- Final validation accuracy: **92.50%**
- Initial training loss: **1.2484**
- Final training loss: **0.0191**
- Final validation loss: **0.2765**
- Approximate training time: **104.1 seconds**

## Observation

Validation accuracy increased from 87.29% during the first epoch to a maximum
of 93.12%. However, it did not exceed the 93.33% achieved by Experiment 1.

Training loss decreased to 0.0191. This was higher than the 0.0059 training
loss observed without augmentation, which is consistent with the augmented
training task being more difficult.

Despite this additional regularization, validation accuracy did not improve.

## Interpretation

The augmentation strategy did not provide a measurable validation improvement
under the tested configuration.

The difference between 93.12% and 93.33% is only 0.21 percentage points, so I
do not interpret this experiment as evidence that data augmentation is
generally harmful. Rather, the specific combination and strength of
RandomResizedCrop and horizontal flipping did not improve this model under the
current training conditions.

## Decision

I rejected this augmentation configuration as an improvement.

Experiment 1 remained the strongest configuration with **93.33% validation
accuracy**.

This negative result motivated a more targeted optimization experiment rather
than simply increasing augmentation strength.


# Experiment 3 — Cosine Learning-Rate Scheduling

## Hypothesis

Experiment 1 achieved high validation accuracy very early in training and then
plateaued. I studied learning-rate scheduling to understand whether reducing
the learning rate gradually during fine-tuning could allow smaller parameter
updates and improve the pretrained model after its initial rapid convergence.

The hypothesis was that cosine learning-rate decay might improve validation
performance during later epochs without changing the architecture or
preprocessing pipeline.

## Controlled Changes

To isolate the effect of the learning-rate schedule, Experiment 3 returned to
the non-augmented preprocessing used in Experiment 1.

The following settings were retained:

- ImageNet-pretrained ResNet18
- 224 × 224 RGB input
- ImageNet normalization
- Adam optimizer
- Initial learning rate: 0.0001
- CrossEntropyLoss
- Batch size: 64
- 10 epochs
- Random seed: 0
- Same training/validation indices

The primary change was:

- CosineAnnealingLR
- T_max = 10
- Minimum learning rate = 0.000001

## Results

- Best validation accuracy: **93.13%**
- Best epoch: **5**
- Final validation accuracy: **92.29%**
- Initial training loss: **1.2511**
- Final training loss: **0.0126**
- Final validation loss: **0.2295**
- Approximate training time: **97.1 seconds**

## Observation

The learning rate gradually decreased during training. Validation accuracy
increased from 88.96% in epoch 1 to a maximum of 93.13% at epoch 5.

However, the scheduled model did not exceed the 93.33% validation accuracy
obtained by Experiment 1.

Training loss continued decreasing throughout training, while validation
accuracy remained approximately 92–93%, indicating that reducing the learning
rate alone did not eliminate the remaining generalization gap.

## Interpretation

Cosine learning-rate scheduling provided a controlled fine-tuning process but
did not produce a measurable improvement over the fixed learning rate.

The difference between Experiment 1 (93.33%) and Experiment 3 (93.13%) was
only 0.20 percentage points. Therefore, I do not conclude that learning-rate
scheduling is generally ineffective. Rather, this specific schedule did not
improve performance under the tested configuration.

## Decision

I rejected Experiment 3 as the final configuration because it did not exceed
Experiment 1.

Experiment 1 remained the selected model with **93.33% validation accuracy**.
It also provided a simpler optimization procedure because no learning-rate
scheduler was required.


# Experiment 4 — Increasing CNN Capacity with ResNet50

## Hypothesis

Experiments 1–3 showed that the ImageNet-pretrained ResNet18 plateaued near
93% validation accuracy. Neither moderate data augmentation nor cosine
learning-rate scheduling improved this result substantially.

I studied the effect of model capacity and investigated whether a deeper
residual CNN could learn more discriminative visual representations for the
16-class classification problem.

I therefore replaced ResNet18 with ImageNet-pretrained ResNet50 while keeping
the 224 × 224 RGB input pipeline and other major training settings unchanged.

## Configuration

- Architecture: ResNet50
- Pretraining: ImageNet
- Input resolution: 224 × 224 RGB
- Data augmentation: None
- Optimizer: Adam
- Learning rate: 0.0001
- Loss: CrossEntropyLoss
- Epochs: 10
- Trainable parameters: 23,540,816

## Results

- Best validation accuracy: **94.37%**
- Best epoch: **7**
- Final epoch validation accuracy: **94.17%**
- Final training loss: **0.0060**
- Final validation loss: **0.2772**
- Training time: approximately **245 seconds**

## Observation

ResNet50 achieved 94.37% validation accuracy, improving upon the 93.33%
obtained with ResNet18. This indicates that increasing CNN capacity provided
a measurable improvement.

However, the training and validation loss curves reveal a generalization gap.
Training loss continued decreasing to nearly zero, while validation loss
reached its lowest region earlier and subsequently increased. Validation
accuracy also plateaued around 94% rather than continuing to improve.

## Interpretation

The improvement from ResNet18 to ResNet50 suggests that additional model
capacity is useful for this dataset. However, the near-zero training loss
combined with increasing validation loss indicates overfitting. Simply
increasing network depth was therefore not sufficient to achieve the strongest
performance.

## Decision

ResNet50 became the current best-performing experiment at **94.37% validation
accuracy**. The next experiment investigated an alternative pretrained CNN
architecture rather than simply increasing network depth further.


# Experiment 5 — Modern CNN Architecture: ConvNeXt-Tiny

## Hypothesis

Experiment 4 showed that increasing model capacity from ResNet18 to ResNet50
improved validation accuracy from 93.33% to 94.37%. However, the ResNet50
training curves still showed a considerable generalization gap.

I studied modern CNN architectures and learned that ConvNeXt modernizes
convolutional neural networks using architectural design choices intended to
provide stronger visual representations while retaining a convolutional
architecture.

I therefore investigated whether an ImageNet-pretrained ConvNeXt-Tiny could
provide stronger transferable visual features for this 16-class scene
classification problem.

To make the comparison controlled, I retained the same 224 × 224 RGB input,
training/validation split, Adam optimizer, learning rate, CrossEntropyLoss,
and 10-epoch training budget. The primary experimental change was the CNN
architecture.

## Configuration

- Architecture: ConvNeXt-Tiny
- Pretraining: ImageNet
- Input resolution: 224 × 224 RGB
- Data augmentation: None
- Optimizer: Adam
- Learning rate: 0.0001
- Loss: CrossEntropyLoss
- Epochs: 10

## Results

- Best validation accuracy: **97.08%**
- Best epoch: **2**
- Final validation accuracy: **96.04%**
- Final training loss: **0.0047**
- Final validation loss: **0.1537**
- Approximate training time: **326.3 seconds**

## Observation

ConvNeXt-Tiny produced a substantial improvement over the previous CNN
architectures. Validation accuracy reached 97.08% at epoch 2, compared with
94.37% for ResNet50 and 93.33% for ResNet18.

The validation loss also decreased considerably. However, after the early
validation peak, training loss continued decreasing toward zero while
validation accuracy fluctuated around 96%.

## Interpretation

The improvement indicates that architecture choice had a larger effect than
the augmentation and learning-rate scheduling changes tested previously.
ConvNeXt-Tiny transferred particularly well to this dataset.

The model also converged extremely quickly. Its best validation result occurred
at epoch 2, after which additional optimization improved training loss but did
not improve validation accuracy. This suggests that later training primarily
increased fitting to the training data rather than improving generalization.

## Decision

ConvNeXt-Tiny became the strongest model with **97.08% validation accuracy**.
The following experiments retained ConvNeXt-Tiny and investigated whether
regularization or additional image information could further improve
generalization.


# Experiment 6 — ConvNeXt-Tiny with Label Smoothing

## Hypothesis

Experiment 5 achieved 97.08% validation accuracy, but training loss became
extremely small while validation performance plateaued. This suggested that
the model was becoming highly confident on the training set.

I studied label smoothing as a regularization technique and investigated
whether reducing overconfident predictions could improve validation
performance.

## Configuration

- Architecture: ConvNeXt-Tiny
- Pretraining: ImageNet
- Input resolution: 224 × 224
- Input channels: RGB
- Training augmentation: None
- Optimizer: Adam
- Learning rate: 0.0001
- Loss: CrossEntropyLoss with label smoothing
- Label smoothing: 0.1
- Epochs: 10

## Implementation Verification

During the initial implementation, the results appeared unexpectedly similar
to Experiment 5. I therefore inspected the training pipeline rather than
assuming that the new configuration was active.

The training function and criterion argument were checked, and the corrected
run explicitly passed the label-smoothed CrossEntropyLoss to the training
function. A diagnostic check confirmed an active label smoothing value of 0.1.

## Results

- Best validation accuracy: **96.88%**
- Best epoch: **3**
- Best validation loss: approximately **0.2013**
- Training loss at epoch 10: **0.5688**
- Validation accuracy at epoch 10: **96.25%**
- Approximate training time: **339.6 seconds**

## Observation

Label smoothing substantially changed the optimization behavior. Unlike
Experiment 5, where training loss approached zero, the training loss remained
around 0.57 because the targets were intentionally softened.

However, the best validation accuracy decreased slightly from 97.08% in
Experiment 5 to 96.88% in Experiment 6.

## Interpretation

Label smoothing successfully reduced model confidence but did not improve
validation accuracy under the tested configuration.

The implementation check was also important: the suspicious initial behavior
showed why experimental changes must be verified in the actual training
pipeline rather than assumed to be active simply because code executes.

## Decision

Label smoothing was not retained. Experiment 5 remained the strongest
configuration.


# Experiment 7 — Higher Input Resolution

## Hypothesis

Increasing the input resolution from 224 × 224 to 320 × 320 might preserve
additional spatial details and help the CNN distinguish visually similar scene
categories.

## Configuration

- Architecture: ConvNeXt-Tiny
- Pretraining: ImageNet
- Input resolution: 320 × 320 RGB
- Augmentation: None
- Optimizer: Adam
- Learning rate: 0.0001
- Loss: CrossEntropyLoss
- Label smoothing: 0.0
- Epochs: 10
- Training images: 1,920
- Validation images: 480
- Same train/validation split as previous experiments

## Results

- Best validation accuracy: **96.46%**
- Best epoch: **2**
- Experiment 5 validation accuracy: **97.08%**
- Change relative to Experiment 5: **-0.62 percentage points**
- Approximate training time: **703 seconds**

## Observation

Increasing image resolution from 224 × 224 to 320 × 320 did not improve
validation performance. Validation accuracy peaked early and then declined,
while training loss continued to approach zero.

The higher-resolution experiment also required substantially more computation
than Experiment 5.

## Interpretation

Additional spatial resolution did not provide useful information sufficient to
improve classification performance on this dataset. The experiment instead
increased computational cost while slightly decreasing validation accuracy.

## Decision

I rejected 320 × 320 as the final input resolution and returned to 224 × 224.


# Experiment 8 — ConvNeXt-Tiny with Mild Data Augmentation

## Hypothesis

Validation error analysis of Experiment 5 showed that several remaining errors
occurred between visually similar scene categories.

I investigated whether mild augmentation could improve generalization without
using the stronger RandomResizedCrop transformation tested earlier with
ResNet18.

The hypothesis was that horizontal flipping and mild color variation could
increase training diversity while preserving the semantic content of the
scene images.

## Configuration

- Architecture: ConvNeXt-Tiny
- Pretraining: ImageNet
- Input resolution: 224 × 224 RGB
- Training augmentation: RandomHorizontalFlip + mild ColorJitter
- Optimizer: Adam
- Learning rate: 0.0001
- Loss: CrossEntropyLoss
- Label smoothing: 0.0
- Epochs: 10
- Same training/validation split as Experiment 5

## Results

- Best validation accuracy: **96.88%**
- Best epoch: **2**
- Experiment 5 validation accuracy: **97.08%**
- Change relative to Experiment 5: **-0.20 percentage points**

## Observation

The augmented model maintained strong validation performance but did not
outperform the simpler Experiment 5 pipeline.

Its best validation accuracy was 96.88%, compared with 97.08% without
training-time augmentation.

## Interpretation

The mild augmentation did not provide additional useful generalization under
the tested ConvNeXt-Tiny configuration.

This does not demonstrate that augmentation is generally ineffective. It shows
that this specific combination of horizontal flipping and mild ColorJitter did
not improve the selected pretrained model on this dataset.

## Decision

The augmentation configuration was rejected as the final model. Experiment 5
remained the best-performing validation configuration.


# Experiment Comparison

| Experiment | Model | Main Change | Best Validation Accuracy | Best Epoch | Observation |
|---|---|---|---:|---:|---|
| 0 | Starter TNet | Original baseline | 49.79% | 11 | Simple baseline strongly overfit |
| 1 | ResNet18 | ImageNet transfer learning + RGB 224×224 | 93.33% | 2 | Transfer learning produced a major improvement |
| 2 | ResNet18 | Added RandomResizedCrop + RandomHorizontalFlip | 93.12% | 4 | Augmentation did not improve the result |
| 3 | ResNet18 | Cosine learning-rate scheduling | 93.13% | 5 | Scheduling did not overcome the ResNet18 plateau |
| 4 | ResNet50 | Increased CNN capacity | 94.37% | 7 | Higher capacity improved performance but overfitting remained |
| 5 | ConvNeXt-Tiny | Changed pretrained CNN architecture | **97.08%** | **2** | **Modern CNN architecture produced the strongest improvement** |
| 6 | ConvNeXt-Tiny | Added label smoothing (0.1) | 96.88% | 3 | Label smoothing slightly reduced validation accuracy |
| 7 | ConvNeXt-Tiny | Increased input resolution to 320×320 | 96.46% | 2 | Higher resolution increased computation without improving validation accuracy |
| 8 | ConvNeXt-Tiny | Added RandomHorizontalFlip + mild ColorJitter | 96.88% | 2 | Additional augmentation did not outperform Experiment 5 |


# Validation Error Analysis

The selected Experiment 5 model correctly classified **466 of 480 validation
images**, corresponding to **97.08% validation accuracy**. Only 14 validation
images were misclassified.

Per-class analysis showed that the remaining errors were concentrated in a
small number of visually related scene categories.

The most difficult class was **Bedroom**, with 33 of 39 images classified
correctly (**84.62%**). Five Bedroom images were predicted as LivingRoom and
one was predicted as Kitchen.

Other notable confusions included:

- Mountain → OpenCountry: 2
- OpenCountry → Mountain: 1
- OpenCountry → Highway: 1
- Store → Kitchen: 1
- InsideCity → Street: 1
- Highway → InsideCity: 1
- Forest → Mountain: 1

Nine of the sixteen classes achieved 100% validation accuracy.

The error pattern suggests that the remaining challenge was primarily
fine-grained discrimination between visually similar scene categories rather
than a broad failure of the learned representation.


# Failure / Unexpected Experiment

Experiment 6 provided both a negative experimental result and an important
implementation lesson.

After Experiment 5 achieved 97.08% validation accuracy, I investigated label
smoothing because the ConvNeXt-Tiny training loss approached zero while
validation performance plateaued. I expected label smoothing to reduce
overconfidence and potentially improve generalization.

The first attempted implementation produced behavior that appeared
unexpectedly similar to Experiment 5. Instead of accepting the result, I
inspected the training pipeline and explicitly verified whether the custom
criterion was actually being used.

The training call was corrected to pass the label-smoothed criterion, and a
diagnostic check confirmed a label smoothing value of 0.1.

After correction, training behavior changed substantially. Final training loss
remained approximately 0.5688 instead of approaching zero, demonstrating that
label smoothing was active. However, best validation accuracy was **96.88%**,
slightly below Experiment 5's **97.08%**.

This experiment showed that reducing prediction confidence did not necessarily
translate into higher classification accuracy for this dataset. More
importantly, it reinforced the need to verify that an experimental change is
actually active rather than relying only on code execution or an apparently
reasonable implementation.


# Final Model Selection

The final model was selected using validation performance only. The test set
was not used to choose the architecture, regularization method, input
resolution, augmentation strategy, or hyperparameters.

Experiment 5 achieved the highest validation accuracy among all tested
configurations.

## Selected Configuration

- Architecture: **ConvNeXt-Tiny**
- Pretraining: **ImageNet**
- Input resolution: **224 × 224 RGB**
- Normalization: **ImageNet mean and standard deviation**
- Training augmentation: **None**
- Optimizer: **Adam**
- Learning rate: **0.0001**
- Loss: **CrossEntropyLoss**
- Epochs: **10**
- Best epoch: **2**
- Best validation accuracy: **97.08%**
- Validation correct predictions: **466 / 480**

The best-validation model state was retained for final evaluation.

## Final Test Evaluation

After model selection was complete, the selected ConvNeXt-Tiny model was
evaluated on the held-out 400-image test set using the same 224 × 224 RGB
preprocessing and ImageNet normalization.

- Test samples: **400**
- Correct predictions: **373**
- Incorrect predictions: **27**
- Final test accuracy: **93.25%**
- Final test loss: **0.2263**

The held-out test result was recorded as the final generalization measurement.
No additional architecture selection or hyperparameter tuning was performed
using the test result.


# Final Conclusion

The experimental process improved validation accuracy from **49.79%** with the
starter CNN to **97.08%** with the selected ImageNet-pretrained ConvNeXt-Tiny.

The largest improvement came from transfer learning, which increased
validation accuracy from 49.79% to 93.33%. Increasing residual-network
capacity to ResNet50 provided a smaller improvement to 94.37%, while switching
to ConvNeXt-Tiny produced the strongest result of 97.08%.

The subsequent controlled experiments showed that additional complexity did
not automatically improve performance. Label smoothing reached 96.88%,
320 × 320 input reached 96.46%, and mild augmentation reached 96.88%.

Therefore, the final system uses the comparatively simple Experiment 5
configuration: an ImageNet-pretrained ConvNeXt-Tiny with 224 × 224 RGB input,
standard CrossEntropyLoss, Adam optimization, and no additional training-time
augmentation.

The selected model achieved **97.08% validation accuracy** and **93.25%
held-out test accuracy**.