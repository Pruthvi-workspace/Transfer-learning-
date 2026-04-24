<h1 align="center">Transfer Learning & Comparative Analysis: ResNet-50 vs. DenseNet-121 on CalTech-101</h1>
<h3 align="center">Standard Fine-Tuning vs. Adaptive Fine-Tuning with ImageNet Pretrained Weights</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-CalTech--101-FFD21E?style=flat&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Models-ResNet50 • DenseNet121-00C853?style=flat" />
  <img src="https://img.shields.io/badge/Accuracy->95%25-00C853?style=flat" />
  <img src="https://img.shields.io/badge/Environment-Google_Colab_GPU-F9AB00?style=flat&logo=googlecolab&logoColor=black" />
</p>

---

## Overview

This project presents a comprehensive transfer learning study comparing **ResNet-50** and **DenseNet-121** on the CalTech-101 dataset using ImageNet-pretrained weights. Two fine-tuning strategies are evaluated:

- **Standard Fine-Tuning (SFT)** — Selective unfreezing of upper layers with differential learning rates.
- **Adaptive Fine-Tuning (AFT)** — A custom strategy that selectively adapts only the most discriminative filters in the final convolutional block using importance-based gradient masking.

All models achieve **>95% Top-1 accuracy**, with DenseNet-121 Standard Fine-Tuning delivering the best overall performance. The study highlights the effectiveness of transfer learning on small datasets, the parameter efficiency of DenseNet, and the strong regularization effect of selective adaptation in ResNet-50.

---

## Key Highlights

- Comparative analysis of ResNet-50 and DenseNet-121 using ImageNet-pretrained weights (IMAGENET1K_V1)
- Implementation of **Standard Fine-Tuning (SFT)** with differential learning rates
- Novel **Adaptive Fine-Tuning (AFT)** strategy based on per-channel activation importance
- Stratified per-class dataset split (70/10/20) to ensure balanced evaluation
- Comprehensive evaluation using Top-1 Accuracy and Macro-F1 score
- Detailed analysis of parameter efficiency, training dynamics, and overfitting behavior
- Loss curves, confusion matrix insights, and computational cost comparison

---

## Architecture Comparison

| Aspect                     | ResNet-50                                      | DenseNet-121                                      |
|----------------------------|------------------------------------------------|---------------------------------------------------|
| Core Mechanism             | Residual connections (additive shortcuts)      | Dense connectivity (channel-wise concatenation)   |
| Key Advantage              | Gradient highway, mitigates vanishing gradients| Feature reuse & implicit deep supervision         |
| Total Parameters           | ~23.7M                                         | ~7.1M                                             |
| Parameter Efficiency       | Lower                                          | 3.4× more efficient than ResNet-50                |
| Best Strategy              | Adaptive Fine-Tuning                           | Standard Fine-Tuning                              |

---

## Dataset & Preprocessing

### CalTech-101
- 101 object categories
- ~9,146 total images (~91 images per class on average)
- Severe long-tail distribution (40–800 images per class)

### Split Strategy
Stratified per-class split (seed=42):
- Train: 70% (~6,402 images)
- Validation: 10% (~914 images)
- Test: 20% (~1,830 images)

### Preprocessing Pipeline
- Grayscale to RGB conversion
- Resize(256) → CenterCrop(224)
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Training augmentations: RandomHorizontalFlip(p=0.5), ColorJitter(±0.2)
- Evaluation: No augmentation

---

## Methodology

### Strategy I — Standard Fine-Tuning (SFT)
- Initial full backbone freeze followed by selective unfreezing
- Differential learning rates: Backbone = 1e-4, Head = 1e-3
- ResNet-50: layer3, layer4, and classifier unfrozen
- DenseNet-121: denseblock3, transition3, denseblock4, norm5, and classifier unfrozen

### Strategy II — Adaptive Fine-Tuning (AFT)
Custom importance-based filter selection:
1. Freeze entire backbone
2. Compute per-channel mean absolute activation on ~3,200 training images
3. Select top 25% most important filters in the final convolutional block
4. Apply binary gradient mask during backpropagation

Target layers:
- ResNet-50: Last Conv2d in layer4 (2048 channels)
- DenseNet-121: conv2 in the last DenseLayer of denseblock4 (growth rate = 32)

### Hyperparameters

| Parameter              | Value                          | Justification                              |
|------------------------|--------------------------------|--------------------------------------------|
| Optimizer              | Adam                           | Stable with mixed learning rates           |
| Backbone LR            | 1e-4 (SFT) / 5e-5 (AFT)        | Prevents catastrophic forgetting           |
| Head LR                | 1e-3                           | Faster convergence for random head         |
| LR Scheduler           | ReduceLROnPlateau (factor=0.5) | Adaptive to validation loss                |
| Weight Decay           | 1e-4                           | Regularization                             |
| Loss                   | CrossEntropy + Label Smoothing (ε=0.1) | Reduces overconfidence               |
| Dropout (head)         | 0.4                            | Prevents co-adaptation                     |
| Batch Size             | 32                             | Fits GPU memory                            |
| Early Stopping         | Patience=7 on val_loss         | Restores best checkpoint                   |

---

## Experimental Results

### Test Set Performance

| Model          | Strategy         | Top-1 Acc. | Macro-F1 | Epochs | Best Val Loss |
|----------------|------------------|------------|----------|--------|---------------|
| ResNet-50      | Standard FT      | 95.52%     | 0.9387   | 25     | 0.9126        |
| DenseNet-121   | Standard FT      | **96.29%** | **0.9504**| 56     | 0.9258        |
| ResNet-50      | Adaptive FT      | 96.18%*    | 0.9482*  | 35     | 0.9519        |
| DenseNet-121   | Adaptive FT      | 95.63%     | 0.9404   | 39     | 0.9878        |

*Notable result: ResNet-50 AFT outperforms its Standard FT counterpart with 6× fewer trainable parameters.

### Model Complexity & Computational Cost

| Model          | Strategy       | Total Params | Trainable Params | GFLOPs | Epoch Time |
|----------------|----------------|--------------|------------------|--------|------------|
| ResNet-50      | Standard FT    | 23.71M       | 22.27M           | ~4.1   | ~14s       |
| DenseNet-121   | Standard FT    | 7.06M        | 5.63M            | ~2.9   | ~14s       |
| ResNet-50      | Adaptive FT    | 23.71M       | ~3.80M           | ~4.1   | ~10.5s     |
| DenseNet-121   | Adaptive FT    | 7.06M        | ~1.40M           | ~2.9   | ~10s       |

DenseNet-121 is **3.4× more parameter-efficient** than ResNet-50. Adaptive Fine-Tuning reduces trainable parameters by ~75% while maintaining strong performance.

---

## Comparative Analysis & Key Insights

- **DenseNet-121 Standard FT** achieves the best overall performance due to superior feature reuse, implicit regularization, and multi-scale feature availability.
- **ResNet-50 Adaptive FT** delivers a compelling efficiency result: 96.18% accuracy using only ~3.8M trainable parameters (6× reduction vs. Standard FT).
- Dense connectivity in DenseNet provides natural internal filter selection, making external gradient masking (AFT) less beneficial compared to ResNet.
- All models show rapid convergence and controlled overfitting thanks to label smoothing, dropout, weight decay, and early stopping.

---

## Training Dynamics

- Training accuracy reaches ~99–100% by epoch 20–25 across all runs.
- Validation accuracy stabilizes between 95–97%.
- ResNet-50 variants converge faster due to cleaner gradient flow via residual connections.
- DenseNet-121 requires more epochs but exhibits smoother validation curves.
- Adaptive Fine-Tuning shows steeper initial loss descent and reduced validation loss fluctuations.

---

## Conclusion

Transfer learning from ImageNet proves highly effective on the small CalTech-101 dataset, with all configurations exceeding **95% Top-1 accuracy**. DenseNet-121 under Standard Fine-Tuning delivers the best accuracy (96.29%) and Macro-F1 (0.9504), benefiting from dense feature reuse and parameter efficiency.

The Adaptive Fine-Tuning strategy demonstrates that **parameter quality matters more than quantity** on limited data — particularly for residual architectures like ResNet-50, where selective gradient masking acts as a powerful regularizer.

Early stopping, label smoothing, and differential learning rates were critical in obtaining reliable, well-generalized results.

---

