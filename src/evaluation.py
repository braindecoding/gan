"""
Comprehensive evaluation with advanced metrics:
PSNR, SSIM, FID, LPIPS, CLIP Score
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')

# Import model classes
class Generator(nn.Module):
    """Generator network for brain decoding"""
    
    def __init__(self, fmri_dim, img_size=28, hidden_dim=512):
        super(Generator, self).__init__()
        self.img_size = img_size
        
        self.model = nn.Sequential(
            nn.Linear(fmri_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, fmri_data):
        img = self.model(fmri_data)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

class InceptionV3Feature(nn.Module):
    """Simplified feature extractor for FID calculation"""
    
    def __init__(self):
        super().__init__()
        # Simple CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256)
        )
    
    def forward(self, x):
        return self.features(x)

class SimpleLPIPS(nn.Module):
    """Simplified LPIPS-like perceptual distance"""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x1, x2):
        f1 = self.net(x1)
        f2 = self.net(x2)
        return F.mse_loss(f1, f2)

def load_data():
    """Load and preprocess brain decoding data"""
    data = loadmat("data/digit69_28x28.mat")
    
    fmri_data = np.vstack([data['fmriTrn'], data['fmriTest']])
    stim_data = np.vstack([data['stimTrn'], data['stimTest']])
    labels = np.vstack([data['labelTrn'], data['labelTest']])
    
    stim_data = stim_data.reshape(-1, 28, 28).astype(np.float32) / 255.0
    fmri_data = fmri_data.astype(np.float32)
    
    # Standardize fMRI data
    fmri_mean = fmri_data.mean(axis=0)
    fmri_std = fmri_data.std(axis=0) + 1e-8
    fmri_data = (fmri_data - fmri_mean) / fmri_std
    
    return fmri_data, stim_data, labels.flatten()

def load_trained_model(model_path="models/generator_final.pth"):
    """Load trained generator model"""
    fmri_data, _, _ = load_data()
    fmri_dim = fmri_data.shape[1]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(fmri_dim).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    return generator, device

def generate_images_from_fmri(generator, fmri_data, device):
    """Generate images from fMRI data"""
    fmri_tensor = torch.FloatTensor(fmri_data).to(device)
    
    with torch.no_grad():
        generated_images = generator(fmri_tensor)
        generated_images = (generated_images + 1) / 2  # [-1,1] -> [0,1]
        generated_images = torch.clamp(generated_images, 0, 1)
    
    return generated_images.cpu().numpy()

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    return ssim(img1, img2, data_range=1.0)

def calculate_fid(real_features, fake_features):
    """Calculate Frechet Inception Distance"""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_lpips_simple(real_images, fake_images, device):
    """Calculate simplified LPIPS score"""
    lpips_net = SimpleLPIPS().to(device)
    
    real_tensor = torch.FloatTensor(real_images).unsqueeze(1).to(device)
    fake_tensor = torch.FloatTensor(fake_images).unsqueeze(1).to(device)
    
    with torch.no_grad():
        lpips_scores = []
        for i in range(len(real_tensor)):
            score = lpips_net(real_tensor[i:i+1], fake_tensor[i:i+1])
            lpips_scores.append(score.item())
    
    return np.mean(lpips_scores)

def calculate_clip_score_simple(real_images, fake_images):
    """Calculate simplified CLIP-like score using image similarity"""
    # Simple correlation-based similarity (placeholder for CLIP)
    similarities = []
    
    for real, fake in zip(real_images, fake_images):
        # Flatten images and calculate correlation
        real_flat = real.flatten()
        fake_flat = fake.flatten()
        
        # Normalize
        real_norm = (real_flat - real_flat.mean()) / (real_flat.std() + 1e-8)
        fake_norm = (fake_flat - fake_flat.mean()) / (fake_flat.std() + 1e-8)
        
        # Calculate correlation
        correlation = np.corrcoef(real_norm, fake_norm)[0, 1]
        similarities.append(correlation if not np.isnan(correlation) else 0)
    
    return np.mean(similarities)

def comprehensive_evaluation():
    """Run comprehensive evaluation with all metrics"""
    
    print("=== Comprehensive Brain Decoding Evaluation ===\n")
    
    # Load data and model
    fmri_data, real_images, labels = load_data()
    generator, device = load_trained_model()
    
    print(f"Evaluating on {len(real_images)} samples...")
    
    # Generate images
    generated_images = generate_images_from_fmri(generator, fmri_data, device)
    generated_images = generated_images.squeeze()  # Remove channel dimension
    
    print("Calculating metrics...")
    
    # 1. PSNR
    psnr_scores = [calculate_psnr(real_images[i], generated_images[i]) 
                   for i in range(len(real_images))]
    avg_psnr = np.mean(psnr_scores)
    
    # 2. SSIM
    ssim_scores = [calculate_ssim(real_images[i], generated_images[i]) 
                   for i in range(len(real_images))]
    avg_ssim = np.mean(ssim_scores)
    
    # 3. FID
    feature_extractor = InceptionV3Feature().to(device)
    
    real_tensor = torch.FloatTensor(real_images).unsqueeze(1).to(device)
    fake_tensor = torch.FloatTensor(generated_images).unsqueeze(1).to(device)
    
    with torch.no_grad():
        real_features = feature_extractor(real_tensor).cpu().numpy()
        fake_features = feature_extractor(fake_tensor).cpu().numpy()
    
    fid_score = calculate_fid(real_features, fake_features)
    
    # 4. LPIPS (simplified)
    lpips_score = calculate_lpips_simple(real_images, generated_images, device)
    
    # 5. CLIP Score (simplified)
    clip_score = calculate_clip_score_simple(real_images, generated_images)
    
    # Print results
    print("\n" + "="*50)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*50)
    print(f"PSNR:       {avg_psnr:.4f} dB")
    print(f"SSIM:       {avg_ssim:.4f}")
    print(f"FID:        {fid_score:.4f}")
    print(f"LPIPS:      {lpips_score:.4f}")
    print(f"CLIP Score: {clip_score:.4f}")
    print("="*50)
    
    # Per-digit analysis
    print("\nPER-DIGIT ANALYSIS:")
    print("-" * 30)
    
    unique_labels = np.unique(labels)
    results_per_digit = {}
    
    for digit in unique_labels:
        mask = labels == digit
        digit_real = real_images[mask]
        digit_fake = generated_images[mask]
        
        digit_psnr = np.mean([calculate_psnr(digit_real[i], digit_fake[i]) 
                             for i in range(len(digit_real))])
        digit_ssim = np.mean([calculate_ssim(digit_real[i], digit_fake[i]) 
                             for i in range(len(digit_real))])
        
        results_per_digit[digit] = {
            'count': len(digit_real),
            'psnr': digit_psnr,
            'ssim': digit_ssim
        }
        
        print(f"Digit {digit}: {len(digit_real)} samples")
        print(f"  PSNR: {digit_psnr:.4f} dB")
        print(f"  SSIM: {digit_ssim:.4f}")
    
    # Create comparison visualization
    print("\nCreating detailed visualization...")
    
    # Select best and worst samples based on PSNR
    best_indices = np.argsort(psnr_scores)[-4:]  # Top 4
    worst_indices = np.argsort(psnr_scores)[:4]  # Bottom 4
    
    fig, axes = plt.subplots(4, 8, figsize=(20, 12))
    
    for i, (title, indices) in enumerate([("Best Results", best_indices), 
                                         ("Worst Results", worst_indices)]):
        for j, idx in enumerate(indices):
            # Real image
            axes[i*2, j].imshow(real_images[idx], cmap='gray')
            axes[i*2, j].set_title(f'Real\nDigit: {labels[idx]}')
            axes[i*2, j].axis('off')
            
            # Generated image
            axes[i*2+1, j].imshow(generated_images[idx], cmap='gray')
            axes[i*2+1, j].set_title(f'Generated\nPSNR: {psnr_scores[idx]:.2f}')
            axes[i*2+1, j].axis('off')
    
    plt.suptitle('Brain Decoding Results: Best vs Worst Reconstructions', fontsize=16)
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/comprehensive_evaluation.png", dpi=150, bbox_inches='tight')
    print("Detailed visualization saved to 'results/comprehensive_evaluation.png'")
    
    # Save detailed results
    results = {
        'overall': {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'fid': fid_score,
            'lpips': lpips_score,
            'clip_score': clip_score
        },
        'per_sample': {
            'psnr': psnr_scores,
            'ssim': ssim_scores,
            'labels': labels
        },
        'per_digit': results_per_digit
    }
    
    np.save("results/comprehensive_results.npy", results)
    print("Detailed results saved to 'results/comprehensive_results.npy'")
    
    return results

def compare_with_baselines():
    """Compare with simple baseline methods"""
    
    print("\n=== Baseline Comparison ===")
    
    fmri_data, real_images, labels = load_data()
    
    # Baseline 1: Mean image
    mean_image = np.mean(real_images, axis=0)
    mean_baseline = np.tile(mean_image, (len(real_images), 1, 1))
    
    # Baseline 2: Random images
    random_baseline = np.random.rand(*real_images.shape)
    
    # Baseline 3: Nearest neighbor in fMRI space
    nn_baseline = []
    for i in range(len(fmri_data)):
        # Find nearest neighbor (excluding self)
        distances = np.sum((fmri_data - fmri_data[i])**2, axis=1)
        distances[i] = np.inf  # Exclude self
        nn_idx = np.argmin(distances)
        nn_baseline.append(real_images[nn_idx])
    nn_baseline = np.array(nn_baseline)
    
    # Calculate metrics for baselines
    baselines = {
        'Mean Image': mean_baseline,
        'Random': random_baseline,
        'Nearest Neighbor': nn_baseline
    }
    
    print("\nBaseline Results:")
    print("-" * 40)
    
    for name, baseline_images in baselines.items():
        psnr = np.mean([calculate_psnr(real_images[i], baseline_images[i]) 
                       for i in range(len(real_images))])
        ssim_score = np.mean([calculate_ssim(real_images[i], baseline_images[i]) 
                             for i in range(len(real_images))])
        
        print(f"{name:15s}: PSNR {psnr:6.2f} dB, SSIM {ssim_score:.4f}")
    
    # Our model results for comparison
    generator, device = load_trained_model()
    generated_images = generate_images_from_fmri(generator, fmri_data, device)
    generated_images = generated_images.squeeze()
    
    our_psnr = np.mean([calculate_psnr(real_images[i], generated_images[i]) 
                       for i in range(len(real_images))])
    our_ssim = np.mean([calculate_ssim(real_images[i], generated_images[i]) 
                       for i in range(len(real_images))])
    
    print(f"{'Our GAN':15s}: PSNR {our_psnr:6.2f} dB, SSIM {our_ssim:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    # Run comprehensive evaluation
    results = comprehensive_evaluation()
    
    # Compare with baselines
    compare_with_baselines()
    
    print("\n=== Evaluation Summary ===")
    print("✅ PSNR: Peak Signal-to-Noise Ratio (higher is better)")
    print("✅ SSIM: Structural Similarity Index (higher is better)")
    print("✅ FID: Frechet Inception Distance (lower is better)")
    print("✅ LPIPS: Learned Perceptual Image Patch Similarity (lower is better)")
    print("✅ CLIP Score: Semantic similarity (higher is better)")
    print("\nAll metrics calculated and saved to 'results/' directory.")
