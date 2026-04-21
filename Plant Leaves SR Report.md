# Plant Leaves Super-Resolution Challenge — Technical Report

## 1. Problem Statement

In modern agricultural technology, automated drones and low-cost mobile sensors photograph crop leaves to detect early signs of disease. Due to hardware limitations, thermal sensor noise, and severe cellular compression, the transmitted images are heavily degraded to 32x32 pixels. Automated diagnostic systems cannot reliably classify pathologies at this resolution.

The objective is to build a Conditional Generative Adversarial Network (cGAN) that performs blind 4x super-resolution, reconstructing 128x128 images from degraded 32x32 inputs. The reconstructed output must recover high-frequency biological textures including leaf veins, chlorosis patterns, and necrotic lesions while remaining mathematically faithful to the ground truth.

The leaderboard evaluation metric is Mean Absolute Error (MAE), computed pixel-by-pixel across all three RGB channels of the flattened 128x128 output (49,152 values per image). Lower MAE corresponds to a higher rank.

### 1.1 Architectural Constraints

- All networks must be randomly initialized. Pretrained weights for the core task are prohibited.
- VGG-19 weights are provided for computing perceptual loss only.
- The inference notebook must run with internet access disabled and no external data.
- The solution must use a GAN-based architecture.

## 2. Dataset

The dataset consists of paired low-resolution and high-resolution plant leaf images in PNG format.

| Split | Count | Resolution |
|---|---|---|
| Training HR (Ground Truth) | 1,642 | 128 x 128 x 3 |
| Training LR (Degraded) | 1,642 | 32 x 32 x 3 |
| Test LR | 495 | 32 x 32 x 3 |

A 90/10 random split produces 1,478 training samples and 164 validation samples. The split is seeded for reproducibility.

Training augmentations applied consistently to both LR and HR images within each pair include random horizontal flip, random vertical flip, and random 90-degree rotation.

## 3. Architecture

### 3.1 Generator

The generator follows the Residual-in-Residual Dense Block (RRDB) architecture. It contains 23 RRDB blocks, each composed of three Residual Dense Blocks with dense skip connections and residual scaling (beta = 0.2). The base channel count is 64 with a growth channel count of 32 per dense layer. Upsampling is performed through two successive PixelShuffle stages, each providing 2x spatial upscaling for a total 4x factor. All activations use LeakyReLU with a negative slope of 0.2. The generator contains approximately 16.8 million trainable parameters.

The generator employs global residual learning. Rather than predicting the full 128x128 output, the generator predicts only the high-frequency residual, which is added to a bicubic upscale of the input:

    SR = Bicubic(LR) + Generator(LR)

This design simplifies the learning task, provides a strong initialization (the model starts at bicubic quality), and accelerates convergence. The final convolutional layer is initialized to zeros so that the initial generator output is the zero residual, meaning the model begins by producing the bicubic baseline.

ICNR initialization is applied to the PixelShuffle convolutional layers to prevent checkerboard artifacts during early training.

The bicubic upscale is computed on CPU using F.interpolate and transferred to GPU. This bypasses a known missing CUDA kernel for bicubic interpolation on Tesla P100 and T4 architectures in PyTorch 2.10.

### 3.2 Discriminator

The discriminator is a conditional PatchGAN. It receives a 6-channel input formed by concatenating the evaluated image (either SR or HR) with the bicubic-upsampled LR input. This conditioning forces the discriminator to judge reconstruction quality relative to each specific input rather than learning dataset-level statistics.

The discriminator consists of four downsampling blocks followed by a single-channel classification head. Spectral normalization is applied to all convolutional layers for Lipschitz continuity. Instance normalization with affine parameters is used in the intermediate blocks. The discriminator contains approximately 2.8 million parameters.

## 4. Training Strategy

### 4.1 Rationale

MAE strictly penalizes spatial inaccuracies. A GAN that generates a realistic texture displaced by a few pixels incurs a large MAE penalty. Therefore, pixel-level structural accuracy must be established before introducing adversarial training. This motivates a two-phase approach.

### 4.2 Phase 1: Pixel-Faithful Regression (250 Epochs)

The generator is trained without the discriminator using three loss functions:

| Loss Function | Weight | Description |
|---|---|---|
| Charbonnier | 1.0 | Differentiable approximation to L1 with smoother gradients near zero. Computed as sqrt((x - y)^2 + epsilon^2). |
| VGG-19 Perceptual | 0.02 | L1 distance between conv5_4 features of the SR and HR images, using the provided VGG-19 weights. |
| FFT | 0.05 | L1 distance between the magnitudes of the 2D real FFT spectra of SR and HR, encouraging high-frequency detail recovery. |

The optimizer is Adam with beta values (0.9, 0.999) and an initial learning rate of 2e-4. A 5-epoch linear warmup precedes cosine annealing to a minimum learning rate of 1e-7. Gradient norms are clipped to a maximum of 1.0.

### 4.3 Phase 2: cGAN Fine-Tuning (100 Epochs)

The best EMA checkpoint from Phase 1 is loaded. The discriminator is activated and the adversarial loss is added to the generator objective.

| Loss Function | Weight |
|---|---|
| Charbonnier | 1.0 |
| VGG-19 Perceptual | 0.02 |
| FFT | 0.05 |
| Adversarial (BCEWithLogits) | 0.001 |

The adversarial weight is set to 0.001, meaning the generator receives 1,000 times more gradient signal from pixel-level losses than from the discriminator. The generator learning rate is reduced to 1e-5 (20x lower than Phase 1) to preserve the established pixel fidelity. The discriminator learning rate is 5e-5. Both use cosine annealing over the 100 Phase 2 epochs.

Label smoothing is applied to the discriminator by setting real labels to 0.9 instead of 1.0, which moderately weakens the discriminator and reduces the risk of mode collapse.

## 5. Stability Mechanisms

### 5.1 Exponential Moving Average

An exponential moving average of all generator parameters is maintained with a decay factor of 0.999. After every optimizer step, the shadow weights are updated:

    shadow = 0.999 * shadow + 0.001 * current_weights

The EMA weights are used for all validation evaluations and final inference. This smooths out training noise, particularly the instability introduced by adversarial training, and produces consistently improving validation metrics.

### 5.2 Gradient Clipping

Gradient norm clipping with a maximum norm of 1.0 is applied to both the generator and discriminator after each backward pass. This prevents gradient explosions that can occur during GAN training.

### 5.3 Spectral Normalization

All convolutional layers in the discriminator use spectral normalization, constraining the spectral norm of each weight matrix to 1. This bounds the Lipschitz constant of the discriminator and stabilizes adversarial training.

## 6. Inference Pipeline

### 6.1 Checkpoint Selection (Smart Ensemble)

EMA checkpoints are saved at epochs 200, 225, 250, 275, 300, 325, and 350, in addition to the overall best EMA checkpoint. Before inference, each checkpoint is evaluated on the validation set. Only checkpoints with validation MAE within 0.5 of the best are retained for ensembling. This automatically excludes any checkpoints degraded by adversarial training in Phase 2.

### 6.2 Test-Time Augmentation

For each test image, 8 augmented versions are processed:

| Fold | Transformation |
|---|---|
| 1 | Identity |
| 2 | 90-degree rotation |
| 3 | 180-degree rotation |
| 4 | 270-degree rotation |
| 5 | Horizontal flip |
| 6 | Horizontal flip + 90-degree rotation |
| 7 | Horizontal flip + 180-degree rotation |
| 8 | Horizontal flip + 270-degree rotation |

Each augmented input is passed through the generator with global residual addition, the inverse transformation is applied to the output, and all 8 predictions are averaged at the pixel level. This reduces directional bias and prediction variance.

### 6.3 Ensemble Averaging

The final prediction for each test image is the pixel-level mean across all selected checkpoints and all 8 TTA folds. If N checkpoints are selected, each test image undergoes N x 8 forward passes. The averaged result is clamped to [0, 1], scaled to [0, 255], and rounded to the nearest integer.

## 7. Submission Format

The output CSV file contains 495 rows (one per test image) with two columns: the image filename and a space-separated string of 49,152 unsigned 8-bit integers representing the flattened 128x128x3 RGB array.

## 8. Results

### 8.1 Phase 1 Convergence (v2 Run, T4 GPU)

Phase 1 training showed stable, monotonic convergence over 200 epochs. Validation MAE decreased from 73.3 at epoch 1 to 17.462 at epoch 200. The EMA validation MAE showed consistent improvement, with nearly every epoch setting a new best. The train-validation MAE gap was 0.475 at convergence, indicating no overfitting and room for additional model capacity.

### 8.2 Phase 2 Observations (v2 Run)

In the v2 run with W_ADV = 0.005, Phase 2 caused the EMA validation MAE to increase from 17.462 to 21.217 over 100 epochs. The adversarial loss drove the generator to hallucinate textures at incorrect spatial locations, resulting in severe MAE penalties. This confirmed the need for a lower adversarial weight in v3.

### 8.3 Leaderboard Performance

The v2 submission achieved a leaderboard MAE of 16.33. This score was obtained despite 5 of the 9 ensemble checkpoints being contaminated by Phase 2 degradation. The TTA and ensemble averaging reduced the effective MAE by approximately 1.1 points from the best single-checkpoint validation MAE.

### 8.4 v3 Architectural Improvements

| Improvement | Mechanism | Expected Impact |
|---|---|---|
| Global Residual Learning | Generator predicts residual from bicubic | 1.0 to 2.0 MAE reduction |
| Smart Ensemble Filtering | Exclude checkpoints beyond 0.5 MAE of best | 0.5 to 1.0 MAE reduction |
| FFT Loss | Frequency-domain L1 on Fourier magnitude | 0.2 to 0.5 MAE reduction |
| 23 RRDB Blocks (from 16) | Increased generator capacity | 0.3 to 0.5 MAE reduction |
| ICNR Initialization | Artifact-free PixelShuffle starting point | 0.1 to 0.3 MAE reduction |
| W_ADV reduced to 0.001 | Prevents Phase 2 MAE degradation | Stabilization |
| G_LR Phase 2 reduced to 1e-5 | Preserves Phase 1 pixel fidelity | Stabilization |

## 9. Computational Requirements

| Component | Time |
|---|---|
| Phase 1 (250 epochs at ~65 sec/epoch) | ~4.5 hours |
| Phase 2 (100 epochs at ~85 sec/epoch) | ~2.4 hours |
| Inference (TTA + Ensemble, 495 images) | ~30 minutes |
| Total | ~7.5 hours |

All training and inference run on a single NVIDIA Tesla T4 GPU (15.6 GB VRAM) within the 12-hour Kaggle session limit. Peak VRAM usage is approximately 12 GB.

## 10. Compliance

| Constraint | Status | Detail |
|---|---|---|
| Random initialization | Satisfied | Generator uses Kaiming normal and ICNR. Discriminator uses Kaiming normal. No pretrained weights loaded for the core task. |
| VGG-19 usage | Satisfied | Loaded from provided vgg19_weights.pth via torch.load. The flag pretrained=True is not used. |
| Internet access | Satisfied | Notebook metadata sets isInternetEnabled to false. |
| External data | Satisfied | Only competition-provided data is used. |
| GAN architecture | Satisfied | Full cGAN with PatchGAN discriminator, adversarial loss, and conditional input. |
| Submission format | Satisfied | 495 rows, 49,152 integer pixels per row. |

## 11. Output Files

    /kaggle/working/
        submission.csv              Final 495-row submission
        submission_bicubic.csv      Bicubic baseline reference
        best_ema.pth                Best EMA generator checkpoint
        ema_ep200.pth               Ensemble checkpoint, epoch 200
        ema_ep225.pth               Ensemble checkpoint, epoch 225
        ema_ep250.pth               Ensemble checkpoint, epoch 250
        ema_ep275.pth               Ensemble checkpoint, epoch 275
        ema_ep300.pth               Ensemble checkpoint, epoch 300
        ema_ep325.pth               Ensemble checkpoint, epoch 325
        ema_ep350.pth               Ensemble checkpoint, epoch 350
        training_history.csv        Per-epoch training metrics

## 12. References

1. Wang, X. et al. (2018). ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks. ECCV Workshops.
2. Lim, B. et al. (2017). Enhanced Deep Residual Networks for Single Image Super-Resolution. CVPR Workshops.
3. Johnson, J. et al. (2016). Perceptual Losses for Real-Time Style Transfer and Super-Resolution. ECCV.
4. Miyato, T. et al. (2018). Spectral Normalization for Generative Adversarial Networks. ICLR.
5. Blau, Y. and Michaeli, T. (2018). The Perception-Distortion Tradeoff. CVPR.
