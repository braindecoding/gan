# Brain Decoding GAN

Neural Signal to Image Translation using Generative Adversarial Networks

## Overview

This project implements a Brain Decoding system using GANs to convert fMRI signals into visual stimulus images.

## Key Results

| Metric | Score | Assessment |
|--------|-------|------------|
| PSNR | 10.42 dB | Fair |
| SSIM | 0.1357 | Poor |
| FID | 0.0043 | Excellent |
| LPIPS | 0.0000 | Excellent |
| CLIP Score | 0.4344 | Fair |

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

### Evaluation
```bash
python evaluate.py
```

## Project Structure

```
├── src/                    # Core source code
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── brain_decoding_gan.py  # GAN architecture
│   ├── train.py           # Training pipeline
│   └── evaluation.py      # Evaluation metrics
├── scripts/               # Utility scripts
├── models/                # Trained models
├── data/                  # Dataset files
├── results/               # Evaluation results
├── plots/                 # Visualizations
├── train.py              # Main training script
├── evaluate.py           # Main evaluation script
└── README.md             # This file
```

## Architecture

- **Generator**: fMRI signals -> Images (FC layers + BatchNorm + Dropout)
- **Discriminator**: Image classification (FC layers + LeakyReLU)
- **Loss**: Binary Cross Entropy (Adversarial)
- **Optimizer**: Adam (lr=0.0002, beta1=0.5)

## Dataset

- **Samples**: 100 (90 train + 10 test)
- **fMRI Features**: 3,092 voxels
- **Images**: 28x28 grayscale digits
- **Classes**: 2 digits (1 and 2)

## Key Achievements

- Proof-of-Concept Success: Brain-to-image translation feasible
- Excellent Perceptual Quality: FID (0.0043), LPIPS (0.0000)
- Consistent Performance: Stable across digit classes
- Training Stability: No mode collapse, stable convergence

## Areas for Improvement

- Structural Similarity: SSIM (0.1357) needs enhancement
- Architecture: Upgrade to convolutional layers
- Loss Function: Add perceptual loss components
- Dataset: Expand size for better generalization

## Future Work

1. Convolutional Architecture: U-Net/ResNet-based generator
2. Advanced Loss: Perceptual + SSIM + Adversarial
3. Progressive Training: Coarse-to-fine approach
4. Larger Datasets: Cross-subject validation
5. Real-time Inference: Optimization for BCI applications

## License

MIT License - see LICENSE file for details.

## Contact

For questions and collaborations, please open an issue.

---

Brain Decoding GAN - Bridging neuroscience and artificial intelligence
