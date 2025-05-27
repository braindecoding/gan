"""
Evaluate trained brain decoding GAN model
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

# Import model classes from training script
class Generator(nn.Module):
    """Generator network for brain decoding"""
    
    def __init__(self, fmri_dim, img_size=28, hidden_dim=512):
        super(Generator, self).__init__()
        self.img_size = img_size
        
        self.model = nn.Sequential(
            # fMRI processing
            nn.Linear(fmri_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.3),
            
            # Image generation
            nn.Linear(hidden_dim, img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, fmri_data):
        img = self.model(fmri_data)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

def load_data():
    """Load and preprocess brain decoding data"""
    print("Loading brain decoding data...")
    
    data = loadmat("data/digit69_28x28.mat")
    
    # Combine train and test data
    fmri_data = np.vstack([data['fmriTrn'], data['fmriTest']])
    stim_data = np.vstack([data['stimTrn'], data['stimTest']])
    labels = np.vstack([data['labelTrn'], data['labelTest']])
    
    # Reshape stimulus data from flat to 28x28
    stim_data = stim_data.reshape(-1, 28, 28).astype(np.float32) / 255.0
    fmri_data = fmri_data.astype(np.float32)
    
    # Standardize fMRI data (same as training)
    fmri_mean = fmri_data.mean(axis=0)
    fmri_std = fmri_data.std(axis=0) + 1e-8
    fmri_data = (fmri_data - fmri_mean) / fmri_std
    
    print(f"fMRI data: {fmri_data.shape}")
    print(f"Stimulus data: {stim_data.shape}")
    print(f"Labels: {labels.shape}")
    
    return fmri_data, stim_data, labels.flatten()

def load_trained_model(model_path="models/generator_final.pth"):
    """Load trained generator model"""
    
    # Get fMRI dimension from data
    fmri_data, _, _ = load_data()
    fmri_dim = fmri_data.shape[1]
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(fmri_dim).to(device)
    
    # Load trained weights
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    print(f"Loaded trained generator from {model_path}")
    print(f"Model has {sum(p.numel() for p in generator.parameters()):,} parameters")
    
    return generator, device

def generate_images_from_fmri(generator, fmri_data, device):
    """Generate images from fMRI data using trained model"""
    
    # Convert to tensor
    fmri_tensor = torch.FloatTensor(fmri_data).to(device)
    
    # Generate images
    with torch.no_grad():
        generated_images = generator(fmri_tensor)
        
        # Convert from [-1, 1] to [0, 1] for visualization
        generated_images = (generated_images + 1) / 2
        generated_images = torch.clamp(generated_images, 0, 1)
    
    return generated_images.cpu().numpy()

def evaluate_model(num_samples=16):
    """Evaluate the trained model"""
    
    print("=== Brain Decoding GAN Evaluation ===\n")
    
    # Load data and model
    fmri_data, real_images, labels = load_data()
    generator, device = load_trained_model()
    
    # Select random samples for evaluation
    indices = np.random.choice(len(fmri_data), num_samples, replace=False)
    
    sample_fmri = fmri_data[indices]
    sample_real = real_images[indices]
    sample_labels = labels[indices]
    
    # Generate images
    print("Generating images from fMRI data...")
    generated_images = generate_images_from_fmri(generator, sample_fmri, device)
    
    # Create visualization
    print("Creating visualization...")
    
    # Calculate grid size
    grid_cols = 4
    grid_rows = (num_samples + grid_cols - 1) // grid_cols
    
    fig, axes = plt.subplots(3, grid_cols, figsize=(16, 12))
    
    for i in range(min(num_samples, grid_cols)):
        # Real image
        axes[0, i].imshow(sample_real[i], cmap='gray')
        axes[0, i].set_title(f'Real Image\nDigit: {sample_labels[i]}')
        axes[0, i].axis('off')
        
        # Generated image
        gen_img = generated_images[i].squeeze()
        axes[1, i].imshow(gen_img, cmap='gray')
        axes[1, i].set_title(f'Generated Image\nDigit: {sample_labels[i]}')
        axes[1, i].axis('off')
        
        # Difference
        diff = np.abs(sample_real[i] - gen_img)
        axes[2, i].imshow(diff, cmap='hot')
        axes[2, i].set_title(f'Absolute Difference\nMSE: {np.mean(diff**2):.4f}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/brain_decoding_results.png", dpi=150, bbox_inches='tight')
    print("Visualization saved to 'results/brain_decoding_results.png'")
    
    # Calculate metrics
    print("\n=== Evaluation Metrics ===")
    
    # Generate images for all samples
    all_generated = generate_images_from_fmri(generator, fmri_data, device)
    
    # Calculate MSE
    mse_per_sample = np.mean((real_images - all_generated.squeeze())**2, axis=(1, 2))
    avg_mse = np.mean(mse_per_sample)
    
    # Calculate PSNR
    psnr_per_sample = 20 * np.log10(1.0 / np.sqrt(mse_per_sample + 1e-8))
    avg_psnr = np.mean(psnr_per_sample)
    
    # Calculate SSIM (simplified version)
    def simple_ssim(img1, img2):
        mu1, mu2 = img1.mean(), img2.mean()
        sigma1, sigma2 = img1.std(), img2.std()
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        return ssim
    
    ssim_scores = [simple_ssim(real_images[i], all_generated[i].squeeze()) for i in range(len(real_images))]
    avg_ssim = np.mean(ssim_scores)
    
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    # Per-digit analysis
    print(f"\n=== Per-Digit Analysis ===")
    unique_labels = np.unique(labels)
    
    for digit in unique_labels:
        digit_mask = labels == digit
        digit_mse = np.mean(mse_per_sample[digit_mask])
        digit_psnr = np.mean(psnr_per_sample[digit_mask])
        digit_count = np.sum(digit_mask)
        
        print(f"Digit {digit}: {digit_count} samples, MSE: {digit_mse:.6f}, PSNR: {digit_psnr:.2f} dB")
    
    # Save detailed results
    results = {
        'avg_mse': avg_mse,
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'mse_per_sample': mse_per_sample,
        'psnr_per_sample': psnr_per_sample,
        'ssim_per_sample': ssim_scores,
        'labels': labels
    }
    
    np.save("results/evaluation_results.npy", results)
    print("\nDetailed results saved to 'results/evaluation_results.npy'")
    
    return results

def compare_models():
    """Compare different model checkpoints"""
    
    print("\n=== Comparing Model Checkpoints ===")
    
    model_files = [
        "models/generator_epoch_10.pth",
        "models/generator_epoch_20.pth", 
        "models/generator_epoch_30.pth",
        "models/generator_final.pth"
    ]
    
    fmri_data, real_images, labels = load_data()
    
    # Select a few samples for comparison
    test_indices = [0, 10, 20, 30]  # Fixed indices for consistent comparison
    test_fmri = fmri_data[test_indices]
    test_real = real_images[test_indices]
    test_labels = labels[test_indices]
    
    fig, axes = plt.subplots(len(model_files) + 1, len(test_indices), figsize=(16, 20))
    
    # Show real images in first row
    for j, idx in enumerate(test_indices):
        axes[0, j].imshow(test_real[j], cmap='gray')
        axes[0, j].set_title(f'Real\nDigit: {test_labels[j]}')
        axes[0, j].axis('off')
    
    # Generate and show images for each model
    for i, model_file in enumerate(model_files):
        if os.path.exists(model_file):
            # Load model
            fmri_dim = fmri_data.shape[1]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            generator = Generator(fmri_dim).to(device)
            generator.load_state_dict(torch.load(model_file, map_location=device))
            generator.eval()
            
            # Generate images
            generated = generate_images_from_fmri(generator, test_fmri, device)
            
            # Calculate average MSE for this model
            all_generated = generate_images_from_fmri(generator, fmri_data, device)
            avg_mse = np.mean((real_images - all_generated.squeeze())**2)
            
            # Show generated images
            for j in range(len(test_indices)):
                axes[i+1, j].imshow(generated[j].squeeze(), cmap='gray')
                if j == 0:
                    epoch = model_file.split('_')[-1].split('.')[0]
                    axes[i+1, j].set_ylabel(f'Epoch {epoch}\nMSE: {avg_mse:.4f}', rotation=0, ha='right')
                axes[i+1, j].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=150, bbox_inches='tight')
    print("Model comparison saved to 'results/model_comparison.png'")

if __name__ == "__main__":
    # Evaluate the final model
    results = evaluate_model(num_samples=8)
    
    # Compare different checkpoints
    compare_models()
    
    print("\n=== Evaluation completed! ===")
    print("Check the 'results/' directory for visualizations and detailed results.")
