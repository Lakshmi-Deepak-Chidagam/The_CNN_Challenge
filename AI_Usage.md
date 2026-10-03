# AI Usage

## Tool Used

I used AI as an pair-programming assistant during this
assignment. I used it primarily for learning concepts, reviewing PyTorch
implementation patterns, debugging code and preprocessing issues, and improving
documentation.

I did not assume AI-generated code or explanations were correct. I inspected
implementations, verified tensor dimensions and active configurations, examined
training behavior, and used measured validation results to determine whether
changes should be retained.


## Example 1 — Understanding the Baseline

After running the starter CNN, I used ChatGPT to clarify the relationship
between training loss, validation loss, and overfitting.

The baseline achieved **49.79% best validation accuracy**. Training loss
decreased from 2.5592 to 0.1181 while validation performance stopped improving.

I compared the explanation with my actual training results and concluded that
the model was strongly overfitting. Based on this evidence, I decided to
investigate transfer learning rather than simply train the starter model longer.


## Example 2 — Implementing Transfer Learning

After deciding to test transfer learning, I used ChatGPT as a coding assistant
while implementing an ImageNet-pretrained ResNet18 in PyTorch.

AI assistance included implementation patterns for:

- loading pretrained ImageNet weights;
- replacing the classifier with a 16-class output layer;
- converting inputs to RGB and resizing them to 224 × 224;
- applying ImageNet normalization; and
- integrating the model with my training loop.

I verified the implementation before relying on its results. The input had
shape `[batch_size, 3, 224, 224]`, and the classifier produced 16 outputs.

The resulting model achieved **93.33% validation accuracy**, compared with
49.79% for the starter CNN. Based on the measured improvement, I retained
transfer learning for subsequent experiments.


## Example 3 — Implementing Controlled Experiments

I used ChatGPT to review implementation approaches for techniques such as data
augmentation, learning-rate scheduling, label smoothing, and alternative CNN
architectures.

I decided which techniques to evaluate and tested changes separately where
practical so that their effects could be interpreted.

For example, Experiment 3 retained ResNet18 and the same validation protocol
while introducing cosine learning-rate scheduling. It achieved **93.13%**
validation accuracy, compared with **93.33%** using the fixed learning rate.

I therefore concluded from my experimental results that cosine scheduling did
not improve this configuration and did not retain it for the final model.


## Example 4 — Debugging Final Test Preprocessing

During final evaluation, I encountered a runtime error because the starter test
DataLoader produced grayscale 64 × 64 images while the selected ConvNeXt-Tiny
expected three-channel RGB input.

I used ChatGPT as a debugging assistant to inspect the error and review the
required preprocessing. I then corrected the test pipeline to use RGB images,
224 × 224 resolution, and ImageNet normalization.

I verified the correction using an actual test batch with shape:

    [32, 3, 224, 224]

I also confirmed that the test loader contained **400 images across 16
classes** before running the final evaluation. The selected model achieved
**93.25% test accuracy**.


## Example 5 — Supporting Validation Error Analysis

After generating predictions from the selected ConvNeXt-Tiny model, I used
ChatGPT to help organize the errors into a concise summary. The predictions and
experimental interpretation came from my model outputs.

Experiment 5 correctly classified **466 of 480 validation images**. The most
frequent confusion was Bedroom → LivingRoom (five cases), with additional
errors involving visually similar categories such as Mountain and OpenCountry.

Based on these observed errors, I decided to test mild augmentation in
Experiment 8. It achieved **96.88% validation accuracy**, below Experiment 5's
97.08%, so I did not retain the augmentation.


# Incorrect / Ineffective AI-Assisted Implementation and Verification

One important verification issue occurred during Experiment 6 while I was using
ChatGPT for assistance with the implementation of label smoothing.

The first implementation produced behavior unexpectedly similar to Experiment
5. Rather than assuming the code was correct because it executed successfully,
I inspected the training pipeline and found that the intended custom criterion
was not being passed to the training function as expected.

I corrected the call so that the label-smoothed criterion was explicitly used
and added a diagnostic check confirming:

    LABEL SMOOTHING: 0.1

After the correction, training behavior changed substantially. Training loss
remained around **0.57** rather than approaching zero, providing additional
evidence that label smoothing was active.

The corrected experiment achieved **96.88% best validation accuracy**, below
Experiment 5's **97.08%**.

I therefore rejected label smoothing based on the corrected experimental
result. This demonstrated why I verified AI-assisted code rather than assuming
that executable code correctly implemented the intended experiment.


# Important Decisions

I was responsible for the architectural and experimental decisions throughout
the project. The most important decision was selecting the final model using
**validation performance rather than test performance**.

After establishing the baseline, I evaluated pretrained ResNet18, augmentation,
learning-rate scheduling, ResNet50, and ConvNeXt-Tiny through controlled
experiments. Experiment 5 achieved the highest validation accuracy at
**97.08%**. Experiments 6, 7, and 8 achieved 96.88%, 96.46%, and 96.88%,
respectively.

Based on these results, I selected Experiment 5 before evaluating the held-out
test set. It achieved **93.25% test accuracy (373/400 correct)**. I did not use
the lower test accuracy to resume hyperparameter tuning or select another
model.

I also chose to retain the simpler 224 × 224 ConvNeXt-Tiny configuration.
Although label smoothing, higher input resolution, and augmentation were
reasonable techniques to investigate, none improved validation accuracy.
Therefore, they were not included in the final system.


# Verification of AI-Assisted Work

I verified AI-assisted work by:

1. Inspecting suggested code before and after execution.
2. Checking training, validation, and test dataset sizes.
3. Verifying input tensor dimensions and RGB preprocessing.
4. Confirming that the classifier produced 16 outputs.
5. Preserving the same train/validation protocol for comparisons.
6. Examining training and validation behavior.
7. Comparing measured validation accuracy across experiments.
8. Confirming that experimental settings and loss functions were active.
9. Investigating suspicious outputs rather than assuming they were correct.
10. Selecting the final model using validation rather than test performance.
11. Checking reported numerical results against the actual experiment outputs.


# Reflection

ChatGPT was useful as a **pair programmer** for learning, implementation,
debugging, and documentation, but it did not replace my responsibility for the
research process.

I made the architectural choices, designed and selected the experiments,
verified AI-assisted implementations, interpreted the measured results, and
selected the final model.

The label-smoothing issue was particularly important because it demonstrated
that AI-assisted code can execute successfully without implementing the
intended experiment correctly. Verification of the active configuration and
actual training behavior was therefore essential before drawing conclusions.