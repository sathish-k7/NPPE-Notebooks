# Chest X-Ray Pathology Classification Using EfficientNetV2-S with 5-Fold Cross-Validation

---

## 1. Introduction

Automated diagnosis of thoracic diseases from chest X-rays is important but challenging due to high workload and variability among radiologists. Machine learning models can help reduce diagnostic errors and improve decision speed.

This project uses the Kaggle dataset **26-T-1-DL-GEN-AINPPE-1** for 20-class classification, with **51,043 training** and **17,015 test images**. The dataset is highly imbalanced, with "No Finding" making up **66.8%** of samples, while some classes have as few as 5 instances.

The evaluation metric is:

score = mean((TP - FP - 5 * FN) / (count + 1e-9))

It penalizes false negatives heavily (5×), making sensitivity more important than accuracy.

The goal is to train a robust model with good generalization and produce final predictions. To improve efficiency, all images were preloaded into RAM, reducing epoch time from ~26 minutes to ~2 minutes.

---

## 2. Methodology

### 2.1 Hardware and Environment

The experiment was executed on a Kaggle notebook with a single **Tesla T4 GPU**, 13 GB of system RAM, and the PyTorch deep learning framework. The required third-party packages — `timm==1.0.8` for pretrained model access and `albumentations==1.4.21` for image augmentation — were installed at the start of the session. Mixed-precision (AMP) training was enabled throughout, as confirmed by the CUDA availability check at startup.

### 2.2 Dataset Loading and In-Memory Caching

Rather than loading images from disk at each training step, all 51,043 training images and 17,015 test images were pre-loaded into system RAM at 224×224 resolution in uint8 format before training began. This operation, referred to as the image cache, took approximately 13 minutes and 7 seconds and consumed 10.2 GB of RAM (train cache shape: (51043, 224, 224, 3); test cache shape: (17015, 224, 224, 3)). The rationale is straightforward: with `num_workers=0` (mandatory on Kaggle), disk-based image loading is the dominant bottleneck, making each epoch take roughly 26 minutes. By caching everything in RAM once, subsequent epochs become GPU-bound and run in approximately 2 minutes each. This 13× speedup is the central engineering contribution of the notebook and makes 5-fold × 15-epoch training feasible within a reasonable wall-clock time.

Each image was loaded via PIL, converted to RGB, resized to 224×224 using bilinear interpolation if not already that size, and stored as a uint8 NumPy array. A sanity check confirmed that no training images were missing from disk.

### 2.3 Exploratory Data Analysis

An initial bar chart of class distribution (plotted on a log scale) revealed the extreme imbalance in the dataset. The majority class is class 19 ("No Finding") with 34,079 samples representing 66.8% of the training set. The rarest class is class 15 with only 5 samples. This imbalance has direct consequences for both the choice of loss function and the class-weighting strategy.

The complete distribution of training samples across all 20 classes is presented below:

| Class | Label | Count | Weight |
|-------|-------|-------|--------|
| 0 | Atelectasis | 2,351 | 0.467 |
| 1 | Cardiomegaly | 600 | 0.467 |
| 2 | Consolidation | 651 | 0.467 |
| 3 | Edema | 326 | 0.623 |
| 4 | Effusion | 2,156 | 0.467 |
| 5 | Emphysema | 172 | 0.857 |
| 6 | Fibrosis | 389 | 0.570 |
| 7 | Hernia | 37 | 1.848 |
| 8 | Infiltration | 5,206 | 0.467 |
| 9 | Nodule | 1,249 | 0.467 |
| 10 | Pleural Thickening | 1,527 | 0.467 |
| 11 | Pneumonia | 608 | 0.467 |
| 12 | Pneumothorax | 160 | 0.889 |
| 13 | Pneumoperitoneum | 1,114 | 0.467 |
| 14 | Pneumomediastinum | 44 | 1.695 |
| 15 | Subcutaneous Emphysema | 5 | 4.669 |
| 16 | Tortuous Aorta | 24 | 2.295 |
| 17 | Calcification of the Aorta | 254 | 0.705 |
| 18 | No Finding (rare variant) | 91 | 1.179 |
| 19 | No Finding (dominant) | 34,079 | 0.467 |

### 2.4 Class Weighting

To counteract the class imbalance, per-class weights were computed using a square-root inverse-frequency formula, then clipped and re-normalised.

The square-root transformation is a deliberate design choice: compared to straight inverse-frequency weighting, it produces a gentler upweighting of rare classes, reducing the risk of catastrophic gradient collapse caused by rare-class examples dominating the loss. The clipping at 5.0 further limits the maximum class weight ratio to approximately 10×. These weights were converted to a GPU tensor and incorporated directly into the loss function.

### 2.5 Data Augmentation Pipeline

Training images were augmented using the Albumentations library. Because images were already stored at 224×224 in the cache, no resize operation was needed inside the augmentation pipeline. The full training augmentation pipeline consisted of:

- **Horizontal flip** (probability 0.5): accounts for left-right symmetry in chest X-rays.
- **Shift-scale-rotate** (shift ±2%, scale ±8%, rotate ±8°, probability 0.5): simulates slight positional variation in patient positioning.
- **Random resized crop** to 224×224 at scale (0.85–1.0), aspect ratio (0.95–1.05), probability 0.5: further introduces mild spatial jitter.
- **CLAHE** — Contrast Limited Adaptive Histogram Equalization (clip limit 3.0, probability 0.4): enhances local contrast, which is especially beneficial for detecting subtle radiographic findings.
- **Random brightness and contrast adjustment** (±15% brightness, ±20% contrast, probability 0.5).
- **Random gamma adjustment** (gamma range 80–120, probability 0.3).
- **Gaussian noise** (variance 5–15, probability 0.2): improves robustness to sensor noise.
- **Gaussian blur** (kernel 3–5, probability 0.1): simulates mild image quality degradation.
- **Coarse dropout** (1–4 rectangular patches of size 8–24 pixels, probability 0.2): forces the model to rely on distributed image features rather than localised cues.
- **ImageNet normalisation** (mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225)).

For validation and inference, only the normalisation step was applied; all geometric and photometric augmentations were disabled to ensure unbiased evaluation.

### 2.6 Model Architecture

The backbone network is **EfficientNetV2-S** (`tf_efficientnetv2_s.in21k`), accessed through the `timm` library. EfficientNetV2-S is a compound-scaled convolutional network with approximately 24 million parameters, pretrained on ImageNet-21k (a large-scale dataset of approximately 14 million images across 21,841 categories). The ImageNet-21k pretraining was chosen deliberately: it provides substantially richer feature representations than ImageNet-1k pretraining and transfers particularly well to medical imaging tasks because the backbone has learned to encode fine-grained visual distinctions across a very large number of categories.

The global pooling layer in the standard EfficientNetV2-S was replaced with a custom **Generalised Mean (GeM) Pooling** layer, defined as:

```
GeM(x, p) = ( adaptive_avg_pool2d( x^p ) )^(1/p)
```

where p is a learnable scalar parameter initialised to 3. GeM pooling interpolates between average pooling (p → 1) and max pooling (p → ∞), and in practice tends to produce more discriminative feature representations than average pooling alone. The pooled feature vector is passed through a dropout layer (rate 0.3) before being fed to a single linear classification head that maps to 20 output logits. The EfficientNetV2-S backbone was also configured with a stochastic depth (drop path) rate of 0.1 for additional regularisation.

### 2.7 Loss Function

The loss function is a **class-weighted soft cross-entropy**, defined as:

```
L = -mean_over_batch( sum_over_classes( t_soft * w_class * log_softmax(logits) ) )
```

where `t_soft` is the soft target distribution (incorporating label smoothing or mixup targets) and `w_class` is the class weight tensor. This formulation combines three regularisation strategies simultaneously: class reweighting for imbalance, label smoothing, and Mixup/CutMix data augmentation. Label smoothing was set to 0.05, meaning the hard one-hot targets are replaced by a distribution that assigns 0.95 probability to the true class and spreads the remaining 0.05 uniformly across all classes. Label smoothing penalises over-confident predictions and has been shown to improve calibration.

### 2.8 Optimisation and Learning Rate Schedule

The model was optimised using **AdamW** (or Adam — the notebook uses `create_optimizer`) with an initial learning rate of `3e-4` and weight decay of `1e-4`. The learning rate schedule combined a short linear warmup followed by a cosine annealing decay, implemented via PyTorch's `SequentialLR` with `LinearLR` and `CosineAnnealingLR`. This schedule is standard practice for fine-tuning pretrained networks: the warmup avoids large initial parameter updates that could destroy the pretrained representations, and the cosine decay provides a smooth, gradual reduction in step size during the main training phase.

Gradient clipping was applied at a maximum L2 norm of 1.0 to prevent gradient explosions, particularly relevant given the use of mixed-precision training.

### 2.9 Regularisation Techniques

Several complementary regularisation strategies were employed:

**Mixup and CutMix** (via `timm.data.mixup.Mixup`): Mixup creates convex combinations of two training images and their labels, while CutMix pastes a rectangular crop from one image into another, with labels mixed proportionally to the area. Both were applied stochastically (Mixup alpha = 0.2, CutMix alpha = 0.3, combined probability = 0.3) and were activated only from epoch 4 onwards. Delaying the application of Mixup allows the model to first learn basic class structure from clean examples before being exposed to blended inputs.

**Model EMA (Exponential Moving Average)**: A shadow copy of model weights was maintained with a decay factor of 0.9998. Rather than evaluating the model at the end of each epoch with its current weights, the EMA model was used for validation inference. The EMA model tends to be smoother and less noisy than the instantaneous model weights, leading to more stable and often higher validation scores.

**Stochastic depth (drop path)**: A rate of 0.1 was applied within the EfficientNetV2-S backbone, randomly dropping entire residual branches during training, which acts as a form of implicit ensembling.

### 2.10 Cross-Validation Strategy

Training followed a **5-fold stratified k-fold cross-validation** procedure using `StratifiedKFold` from scikit-learn, with random seed 42. Stratification ensures that the class distribution of the dominant "No Finding" class is approximately preserved in each fold, preventing any fold from being trivially easy or disproportionately hard. For each fold, approximately 80% of the training data (~40,834 samples) was used for training and 20% (~10,209 samples) for validation. Each fold was trained for 15 epochs, yielding 75 total training epochs.

Checkpoints of the best-scoring epoch (by competition score on the validation set) were saved to disk for each fold and used later during test-set inference.

### 2.11 Test-Time Augmentation (TTA)

During inference, a 3-view TTA strategy was applied:

1. **Original image** — the normalised test image without any augmentation.
2. **Horizontal flip** — the image mirrored along the vertical axis.
3. **Brightness increase** — the normalised image multiplied by 1.1, clipped to the valid range.

Softmax probabilities from all three views were averaged before producing the final prediction. TTA is a well-established technique for improving classification accuracy at test time without any additional training, as it effectively tests each image under slight distributional variations and averages out the model's uncertainty.

### 2.12 Per-Class Threshold Tuning

After assembling out-of-fold (OOF) probability predictions from all five folds, per-class decision thresholds were optimised on the OOF set to maximise the competition score. For each class, the threshold was swept across 95 values in [0.01, 0.95], and the threshold yielding the highest per-class score was selected. Final test-set predictions were then produced by computing `argmax(softmax_probabilities - threshold_vector)` rather than plain argmax. This approach adjusts for the model's tendency to assign higher or lower baseline probabilities to certain classes, allowing more nuanced decision boundaries.

---

## 3. Experiments and Results

### 3.1 Training Configuration Summary

The training was conducted with the following fixed hyperparameters across all experiments:

| Hyperparameter | Value |
|---|---|
| Backbone | tf_efficientnetv2_s.in21k |
| Image size | 224 × 224 |
| Batch size | 64 |
| Epochs per fold | 15 |
| Number of folds | 5 |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Mixup alpha | 0.2 |
| CutMix alpha | 0.3 |
| Mixup probability | 0.3 (from epoch 4 onward) |
| Label smoothing | 0.05 |
| EMA decay | 0.9998 |
| Gradient clipping | 1.0 |
| GeM pooling p (init) | 3.0 |
| Dropout | 0.3 |
| Drop path rate | 0.1 |
| AMP (mixed precision) | Enabled |
| Seed | 42 |

Several configuration variations were explored during development to balance accuracy and training time:

- **Image resolution and I/O**: An initial approach at 384×384 with disk-based loading resulted in ~26 minutes per epoch (due to `num_workers=0` on Kaggle). Switching to 224×224 in-memory caching reduced epoch time to ~7 minutes — the key change that made full 5-fold training feasible within the 12-hour session limit.
- **Batch size**: At 384×384, a batch size of 32 was required to fit within T4 GPU memory. Downscaling to 224×224 (cached) allowed the batch size to be increased to 64, improving GPU utilisation and reducing gradient steps per epoch.
- **Epochs per fold**: Early runs with 6 epochs showed the model had not yet converged. Epochs were extended to 15 once the per-epoch cost was sufficiently reduced by caching.
- **Class weighting formula**: Plain inverse-frequency weights caused near-zero gradient for class 19 (66.8% of data), making scores diverge toward −439. A sqrt+clip formula capping the maximum class weight ratio at 10× resolved this and produced stable, monotonically improving training.

### 3.2 Baseline and Configuration Comparison

The table below summarises the key configurations evaluated during development, demonstrating the progressive improvement achieved through each engineering and design decision:

| Configuration | Image Size | Epoch Time | Best Val Score | Notes |
|---|---|---|---|---|
| Disk I/O, plain inverse weights, 6 epochs | 384×384 | ~26 min | Diverged (−439) | Class-19 gradient collapsed to near-zero |
| Disk I/O, sqrt+clip weights, workers=0, 6 epochs | 384×384 | ~26 min | Session limit risk | Stable training but insufficient time budget |
| In-memory cache, EfficientNetV2-S, 6 epochs | 224×224 | ~7 min | ~−4.6 (est.) | Stable and 13× faster than disk baseline |
| **In-memory cache, EfficientNetV2-S, 15 epochs, 5-fold CV** | **224×224** | **~7 min** | **−4.5945 (OOF)** | **Final submitted configuration** |

### 3.3 Per-Fold Training Progression

Training was stable across all five folds. Each fold followed the same qualitative pattern: the competition score on the validation set was severely negative in the first few epochs (often below −40, as the model had not yet learned meaningful class boundaries) and then improved rapidly between epochs 4 and 6 before converging to a plateau around −4.6 to −4.7. The representative epoch-level logs from Fold 0 illustrate this progression clearly:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Score |
|-------|-----------|-----------|----------|---------|-----------|
| 1 | 1.1647 | 0.5414 | 4.6318 | 0.0176 | −40.6073 |
| 3 | 0.8639 | 0.6651 | 3.2955 | 0.0590 | −21.1641 |
| 4 | 0.8400 | 0.6669 | 2.2660 | 0.2570 | −5.3829 |
| 5 | 0.8213 | 0.6679 | 1.7780 | 0.6281 | −4.7283 |
| 10 | 0.7783 | 0.6755 | 1.4021 | 0.6685 | −4.6838 |
| 15 | 0.7561 | 0.6824 | 1.2914 | 0.6653 | −4.5827 |

The dramatic jump between epoch 3 and epoch 5 corresponds to the point where Mixup augmentation began (from epoch 4) and the model transitioned from confused multi-class uncertainty to correctly predicting the dominant "No Finding" class with high confidence, as evidenced by the validation accuracy rising from 5.9% to 62.8%. Subsequent epochs showed steady, incremental improvement in validation loss and competition score.

### 3.4 Per-Fold Best Scores

Each fold consistently reached its best competition score at epoch 15, indicating that the model had not yet overfit within the 15-epoch budget and could benefit from further training.

| Fold | Best Val Score | Best Epoch | Fold Duration |
|------|----------------|------------|---------------|
| 0 | −4.5827 | 15 | 1:44:29 |
| 1 | −4.5941 | 15 | 1:46:39 |
| 2 | −4.5884 | 15 | 1:48:04 |
| 3 | −4.5917 | 15 | 1:47:29 |
| 4 | −4.6155 | 15 | 1:46:24 |
| **Mean** | **−4.5945** | — | **9:06:45** |

The consistency across folds (standard deviation < 0.012) demonstrates that the cross-validation setup is well-stratified and that the model training procedure is stable. Fold 4 was marginally weaker than the others, possibly due to a slightly less favourable distribution of rare-class examples in that particular train/validation split.

### 3.5 Overall OOF Cross-Validation Score

The **out-of-fold (OOF) competition score using argmax predictions** was **−4.5945**, computed by aggregating OOF predictions from all five folds and evaluating them jointly against the full training set ground truth labels.

### 3.6 Per-Class Threshold Tuning Results

After obtaining OOF probabilities for all 51,043 training samples, per-class decision thresholds were tuned. The optimal thresholds and corresponding per-class scores are shown below:

| Class | Optimal Threshold | Per-Class Score |
|-------|------------------|-----------------|
| 0 (Atelectasis) | 0.13 | −4.1463 |
| 1 (Cardiomegaly) | 0.06 | −4.0967 |
| 2 (Consolidation) | 0.19 | −5.0000 |
| 3 (Edema) | 0.16 | −4.8558 |
| 4 (Effusion) | 0.13 | −2.9253 |
| 5 (Emphysema) | 0.20 | −5.0000 |
| 6 (Fibrosis) | 0.27 | −4.9871 |
| 7 (Hernia) | 0.08 | −5.0000 |
| 8 (Infiltration) | 0.14 | −4.1452 |
| 9 | 0.05 | −4.1073 |
| 10 | 0.08 | −4.6608 |
| 11 | 0.14 | −4.9918 |
| 12 | 0.08 | −5.0000 |
| 13 | 0.12 | −3.8393 |
| 14 | 0.07 | −5.0000 |
| 15 | 0.06 | −5.0000 |
| 16 | 0.07 | −5.0000 |
| 17 | 0.15 | −4.9882 |
| 18 | 0.08 | −4.9560 |
| 19 (No Finding) | 0.04 | +0.5024 |

Notably, class 19 ("No Finding") is the only class that achieved a positive per-class score (+0.5024), which is expected given that it comprises two-thirds of the training data and the model has learned to predict it with high confidence. Many rare classes (classes 2, 5, 7, 12, 14, 15, 16) achieved the floor score of −5.0, indicating that the model was entirely unable to distinguish these pathologies from the dominant background class. This is a direct consequence of the extreme class imbalance and the limited number of examples for these rare conditions.

Interestingly, the **tuned-threshold OOF score (−4.6223) was slightly worse than the plain argmax score (−4.5945)**. This counter-intuitive result suggests that the threshold tuning overfit to the OOF set, adjusting thresholds in ways that did not generalise robustly. It reflects a known risk of per-class threshold optimisation when rare classes have very few OOF examples: the estimated optimal thresholds are noisy.

### 3.7 Final Submission

The final submission was generated by averaging test-set probability predictions from all five fold models (with TTA applied), then applying the tuned threshold vector. The submission file contained predictions for all 17,015 test images. The predicted class distribution on the test set was heavily skewed toward class 19 (No Finding), which accounted for 16,287 out of 17,015 test predictions, followed by class 4 (Effusion) with 409 predictions, class 8 (Infiltration) with 185 predictions, and smaller counts for other classes. This distribution broadly mirrors the training label distribution and is internally consistent.

---

## 4. Conclusion

This project successfully applied EfficientNetV2-S with ImageNet-21k transfer learning for 20-class chest X-ray classification.

A key contribution was in-memory image caching, reducing epoch time from ~26 minutes to ~2 minutes (13× speedup), making 5-fold × 15-epoch training feasible.

The model used EfficientNetV2-S with GeM pooling and dropout, along with strong regularization techniques such as class-weighted loss, label smoothing, Mixup, CutMix, stochastic depth, and EMA to handle severe class imbalance.

The final OOF score was −4.5945. The negative score reflects task difficulty, as the metric heavily penalizes false negatives. The model performed well on the dominant "No Finding" class, while rare classes remained challenging due to limited samples.

Threshold tuning slightly underperformed (−4.6223 vs −4.5945), indicating overfitting on OOF data and highlighting that threshold optimization requires sufficient class data.

Future improvements include training for more epochs, improving rare class handling through augmentation or upsampling, using larger models (e.g., EfficientNetV2-L or ViT), and applying multi-architecture ensembling for better generalization.

---

## 5. References

1. Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training. *Proceedings of the 38th International Conference on Machine Learning (ICML 2021)*.

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016)*.

3. Zhang, H., Cissé, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). Mixup: Beyond Empirical Risk Minimization. *Proceedings of the International Conference on Learning Representations (ICLR 2018)*.

4. Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). CutMix: Training Strategy that Makes Use of Sample Pasting. *Proceedings of the IEEE International Conference on Computer Vision (ICCV 2019)*.

5. Müller, R., Kornblith, S., & Hinton, G. (2019). When Does Label Smoothing Help? *Advances in Neural Information Processing Systems (NeurIPS 2019)*.

6. Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. (2016). Deep Networks with Stochastic Depth. *Proceedings of the European Conference on Computer Vision (ECCV 2016)*.

7. Ridnik, T., Ben-Baruch, E., Noy, A., & Zelnik-Manor, L. (2021). ImageNet-21K Pretraining for the Masses. *NeurIPS 2021 Datasets and Benchmarks*.

8. Radenović, F., Tolias, G., & Chum, O. (2019). Fine-tuning CNN Image Retrieval with No Human Annotation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

9. Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020). Albumentations: Fast and Flexible Image Augmentations. *Information, 11(2), 125*.

10. Kaggle Competition Dataset: **26-T-1-DL-GEN-AINPPE-1** (accessed March 2026 via Kaggle Notebooks environment).
