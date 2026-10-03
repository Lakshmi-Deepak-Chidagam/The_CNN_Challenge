ITCS 6169/8169 Assignment 1 — The CNN Challenge
Lakshmi Deepak Chidagam


1. FINAL RESULT

The dataset contains 2,400 provided training images and 400 held-out test
images across 16 scene classes. I divided the provided training set into
1,920 training and 480 validation images using an 80/20 split with random
seed 0. Validation performance was used for architecture, hyperparameter,
and model-selection decisions. The test set was reserved for evaluation
after the final model was selected.

Final selected model: ImageNet-pretrained ConvNeXt-Tiny
Best validation accuracy: 97.08% (466/480)
Best epoch: 2
Final test accuracy: 93.25% (373/400)
Final test loss: 0.2263

GitHub repository:
https://github.com/Lakshmi-Deepak-Chidagam/The_CNN_Challenge

Final checkpoint:
convnext_tiny_exp5_best.pth

Checkpoint link:
https://drive.google.com/file/d/1uZkkbls349CZjVMuVpd13pMBNsZRyXDE/view?usp=drive_link


2. SECRET RECIPE

The final system uses an ImageNet-pretrained ConvNeXt-Tiny CNN with
224×224 RGB input. Images are normalized using the ImageNet mean and
standard deviation. The original classifier is replaced with a linear
classification layer for the 16 scene categories.

The model is fine-tuned for 10 epochs using Adam with learning rate 1e-4,
batch size 64, standard CrossEntropyLoss, and no additional training-time
augmentation. The best checkpoint is selected according to validation
accuracy; the highest validation result occurred at epoch 2.

The most important improvement was using pretrained CNN representations.
The starter TNet, trained on 64×64 grayscale images, achieved only 49.79%
validation accuracy. An ImageNet-pretrained ResNet18 with 224×224 RGB input
increased this to 93.33%. Increasing CNN capacity with ResNet50 improved the
result to 94.37%, while changing the architecture to ConvNeXt-Tiny produced
the strongest result of 97.08%.

Additional complexity did not improve the selected model. Label smoothing
(0.1) reached 96.88%, increasing input resolution to 320×320 reached 96.46%
while substantially increasing computation, and mild augmentation reached
96.88%. Therefore, I retained the simpler 224×224 ConvNeXt-Tiny configuration.


3. EXPERIMENTAL JOURNEY

Experiments were performed as a sequence of hypotheses and measured using the
same validation protocol where applicable.

Experiment   Model / Main Change                         Best Val.   Epoch
---------------------------------------------------------------------------
0            Starter TNet, 64×64 grayscale                49.79%      11
1            Pretrained ResNet18, 224×224 RGB             93.33%       2
2            ResNet18 + crop/flip augmentation            93.12%       4
3            ResNet18 + cosine LR schedule                93.13%       5
4            Pretrained ResNet50                          94.37%       7
5            Pretrained ConvNeXt-Tiny                     97.08%       2
6            ConvNeXt-Tiny + label smoothing (0.1)        96.88%       3
7            ConvNeXt-Tiny, 320×320 input                 96.46%       2
8            ConvNeXt-Tiny + mild augmentation            96.88%       6

The baseline strongly overfit: training loss decreased from 2.5592 to 0.1181,
but validation accuracy peaked at only 49.79%. Transfer learning produced the
largest improvement, with ResNet18 reaching 93.33%. Neither augmentation nor
cosine learning-rate scheduling improved this result. Increasing capacity to
ResNet50 raised accuracy to 94.37%.

The strongest architecture was ConvNeXt-Tiny, which reached 97.08% at epoch 2.
Training loss continued decreasing afterward while validation accuracy
fluctuated near 96%, indicating that further optimization mainly increased
training-set fit. Experiments 6–8 therefore tested regularization, higher
resolution, and augmentation, but none exceeded Experiment 5.

Validation error analysis also showed that Experiment 5 correctly classified
466/480 images. The largest confusion was Bedroom→LivingRoom (5 cases), with
additional errors primarily occurring between visually similar categories such
as Mountain and OpenCountry.


4. FAILURE ANALYSIS

Experiment 7 tested whether increasing ConvNeXt-Tiny input resolution from
224×224 to 320×320 would preserve additional spatial detail and improve
discrimination between visually similar scene classes. I expected the higher
resolution to potentially help with the remaining fine-grained classification
errors.

Instead, best validation accuracy decreased from 97.08% to 96.46%. Training
time also increased substantially, from approximately 326 seconds for the
224×224 Experiment 5 to approximately 703 seconds. Training loss continued
approaching zero while validation performance did not improve.

This result showed that additional input detail did not provide enough useful
information to justify the increased computational cost. I therefore rejected
320×320 input and retained 224×224 resolution. More generally, this experiment
showed that increasing computational complexity does not necessarily improve
generalization.


5. AI Usage

I used AI as a learning, coding, debugging, and documentation
assistant. I used it to study CNNs, transfer learning, overfitting,
regularization, PyTorch implementation patterns, preprocessing, and controlled
experiment design. AI-assisted implementations were verified using actual
tensor dimensions, active training configurations, learning curves, and
measured validation results.

One useful AI-assisted interaction occurred while debugging final evaluation.
The starter test DataLoader still produced grayscale 64×64 images, whereas
the selected ConvNeXt-Tiny required RGB 224×224 input. I identified the
preprocessing mismatch with AI assistance, corrected the test transform to
224×224 RGB with ImageNet normalization, and verified an actual test batch
shape of [32, 3, 224, 224] before evaluation.

AI output was not assumed to be correct. During Experiment 6, an initial
label-smoothing implementation produced suspiciously similar behavior to
Experiment 5. I inspected the training pipeline and found that the intended
custom criterion was not being passed as expected. I corrected the call and
explicitly verified that label smoothing = 0.1 was active. The corrected run
behaved differently and achieved 96.88% validation accuracy, below the 97.08%
baseline ConvNeXt-Tiny result, so I rejected label smoothing.

An important decision was to select Experiment 5 using validation
performance before evaluating the held-out test set. Although the final test
accuracy (93.25%) was lower than validation accuracy (97.08%), I did not use
the test result to resume hyperparameter tuning or select another model. This
preserved the test set as the final generalization measurement.