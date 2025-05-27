"""
Brain Decoding GAN Architecture
Converts fMRI signals to visual stimulus images
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Generator(nn.Module):
    """
    Generator network that converts fMRI signals to images
    Input: fMRI data (batch_size, fmri_features)
    Output: Generated images (batch_size, channels, height, width)
    """
    
    def __init__(self, fmri_dim: int, img_channels: int = 1, img_size: int = 28, hidden_dim: int = 512):
        super(Generator, self).__init__()
        
        self.fmri_dim = fmri_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        
        # Calculate the size needed for reshaping
        self.init_size = img_size // 4  # Initial size after first upsampling
        
        # fMRI feature processing
        self.fmri_processor = nn.Sequential(
            nn.Linear(fmri_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, self.init_size * self.init_size * 128)
        )
        
        # Convolutional layers for image generation
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            
            # Upsample to 14x14 (for 28x28 final)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Upsample to 28x28
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Final layer
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, fmri_data):
        # Process fMRI data
        out = self.fmri_processor(fmri_data)
        
        # Reshape for convolutional layers
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        
        # Generate image
        img = self.conv_blocks(out)
        
        return img

class Discriminator(nn.Module):
    """
    Discriminator network that distinguishes real from generated images
    Input: Images (batch_size, channels, height, width)
    Output: Probability of being real (batch_size, 1)
    """
    
    def __init__(self, img_channels: int = 1, img_size: int = 28):
        super(Discriminator, self).__init__()
        
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Calculate the size after convolutions
        def conv2d_out_size(size, kernel_size, stride, padding):
            return (size + 2 * padding - kernel_size) // stride + 1
        
        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(img_channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 7x7 -> 3x3
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        
        # Calculate flattened size
        conv_out_size = 3  # After the convolutions
        self.fc_input_size = 128 * conv_out_size * conv_out_size
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.size(0), -1)
        validity = self.fc(out)
        return validity

class BrainDecodingGAN:
    """
    Complete Brain Decoding GAN system
    """
    
    def __init__(self, fmri_dim: int, img_channels: int = 1, img_size: int = 28, 
                 lr: float = 0.0002, beta1: float = 0.5, device: str = None):
        """
        Initialize the Brain Decoding GAN
        
        Args:
            fmri_dim: Dimension of fMRI features
            img_channels: Number of image channels (1 for grayscale, 3 for RGB)
            img_size: Size of generated images (assumed square)
            lr: Learning rate
            beta1: Beta1 parameter for Adam optimizer
            device: Device to run on ('cuda' or 'cpu')
        """
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize networks
        self.generator = Generator(fmri_dim, img_channels, img_size).to(self.device)
        self.discriminator = Discriminator(img_channels, img_size).to(self.device)
        
        # Loss function
        self.adversarial_loss = nn.BCELoss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'epoch': []
        }
    
    def train_step(self, fmri_batch, real_images):
        """
        Single training step
        
        Args:
            fmri_batch: Batch of fMRI data
            real_images: Batch of real stimulus images
            
        Returns:
            Tuple of (generator_loss, discriminator_loss)
        """
        batch_size = fmri_batch.size(0)
        
        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, device=self.device, requires_grad=False)
        fake = torch.zeros(batch_size, 1, device=self.device, requires_grad=False)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        self.optimizer_G.zero_grad()
        
        # Generate images from fMRI data
        generated_images = self.generator(fmri_batch)
        
        # Generator loss
        g_loss = self.adversarial_loss(self.discriminator(generated_images), valid)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        self.optimizer_D.zero_grad()
        
        # Real images loss
        real_loss = self.adversarial_loss(self.discriminator(real_images), valid)
        
        # Fake images loss
        fake_loss = self.adversarial_loss(self.discriminator(generated_images.detach()), fake)
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        self.optimizer_D.step()
        
        return g_loss.item(), d_loss.item()
    
    def generate_images(self, fmri_data):
        """
        Generate images from fMRI data
        
        Args:
            fmri_data: fMRI data tensor
            
        Returns:
            Generated images tensor
        """
        self.generator.eval()
        with torch.no_grad():
            generated_images = self.generator(fmri_data)
        self.generator.train()
        return generated_images
    
    def save_models(self, filepath_prefix: str):
        """Save the trained models"""
        torch.save(self.generator.state_dict(), f"{filepath_prefix}_generator.pth")
        torch.save(self.discriminator.state_dict(), f"{filepath_prefix}_discriminator.pth")
        logger.info(f"Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str):
        """Load pre-trained models"""
        self.generator.load_state_dict(torch.load(f"{filepath_prefix}_generator.pth", map_location=self.device))
        self.discriminator.load_state_dict(torch.load(f"{filepath_prefix}_discriminator.pth", map_location=self.device))
        logger.info(f"Models loaded with prefix: {filepath_prefix}")
    
    def plot_training_history(self):
        """Plot training loss history"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['epoch'], self.history['g_loss'], label='Generator Loss')
        plt.plot(self.history['epoch'], self.history['d_loss'], label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Brain Decoding GAN Training History')
        plt.legend()
        plt.grid(True)
        plt.show()

def visualize_results(gan_model, fmri_data, real_images, num_samples=8):
    """
    Visualize brain decoding results
    
    Args:
        gan_model: Trained BrainDecodingGAN model
        fmri_data: fMRI data tensor
        real_images: Real stimulus images tensor
        num_samples: Number of samples to visualize
    """
    # Generate images
    generated_images = gan_model.generate_images(fmri_data[:num_samples])
    
    # Convert to numpy for visualization
    generated_np = generated_images.cpu().numpy()
    real_np = real_images[:num_samples].cpu().numpy()
    
    # Plot comparison
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Real images
        real_img = real_np[i].squeeze()
        axes[0, i].imshow(real_img, cmap='gray')
        axes[0, i].set_title(f'Real {i+1}')
        axes[0, i].axis('off')
        
        # Generated images
        gen_img = generated_np[i].squeeze()
        axes[1, i].imshow(gen_img, cmap='gray')
        axes[1, i].set_title(f'Generated {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example dimensions (adjust based on your data)
    fmri_dim = 1000  # Number of fMRI features/voxels
    img_channels = 1  # Grayscale images
    img_size = 28     # 28x28 images
    
    # Initialize GAN
    gan = BrainDecodingGAN(fmri_dim, img_channels, img_size)
    
    print("Brain Decoding GAN initialized successfully!")
    print(f"Generator parameters: {sum(p.numel() for p in gan.generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in gan.discriminator.parameters()):,}")
