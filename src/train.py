"""
Simple training script for Brain Decoding GAN without matplotlib
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

class Discriminator(nn.Module):
    """Discriminator network"""
    
    def __init__(self, img_size=28):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class BrainDataset:
    """Simple dataset for brain decoding"""
    
    def __init__(self, fmri_data, stim_data):
        # Standardize fMRI data
        self.fmri = torch.FloatTensor(fmri_data)
        self.fmri = (self.fmri - self.fmri.mean(dim=0)) / (self.fmri.std(dim=0) + 1e-8)
        
        # Normalize stimulus to [-1, 1] for GAN training
        self.stim = torch.FloatTensor(stim_data).unsqueeze(1)  # Add channel dim
        self.stim = (self.stim - 0.5) / 0.5  # [0,1] -> [-1,1]
        
    def __len__(self):
        return len(self.fmri)
    
    def __getitem__(self, idx):
        return self.fmri[idx], self.stim[idx]

def load_data():
    """Load and preprocess brain decoding data"""
    print("Loading brain decoding data...")
    
    data = loadmat("data/digit69_28x28.mat")
    
    # Combine train and test data
    fmri_data = np.vstack([data['fmriTrn'], data['fmriTest']])
    stim_data = np.vstack([data['stimTrn'], data['stimTest']])
    
    # Reshape stimulus data from flat to 28x28
    stim_data = stim_data.reshape(-1, 28, 28).astype(np.float32) / 255.0
    fmri_data = fmri_data.astype(np.float32)
    
    print(f"fMRI data: {fmri_data.shape}")
    print(f"Stimulus data: {stim_data.shape}")
    
    return fmri_data, stim_data

def train_gan(epochs=50, batch_size=16, lr=0.0002):
    """Train the brain decoding GAN"""
    
    # Load data
    fmri_data, stim_data = load_data()
    
    # Create dataset and dataloader
    dataset = BrainDataset(fmri_data, stim_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    fmri_dim = fmri_data.shape[1]
    generator = Generator(fmri_dim).to(device)
    discriminator = Discriminator().to(device)
    
    # Loss and optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"Training on {len(dataset)} samples, {len(dataloader)} batches per epoch")
    
    # Training loop
    print("\nStarting training...")
    
    for epoch in range(epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for fmri_batch, real_images in progress_bar:
            batch_size_actual = fmri_batch.size(0)
            
            # Move to device
            fmri_batch = fmri_batch.to(device)
            real_images = real_images.to(device)
            
            # Adversarial ground truths
            valid = torch.ones(batch_size_actual, 1, device=device, requires_grad=False)
            fake = torch.zeros(batch_size_actual, 1, device=device, requires_grad=False)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            optimizer_G.zero_grad()
            
            # Generate images from fMRI
            generated_images = generator(fmri_batch)
            
            # Generator loss
            g_loss = adversarial_loss(discriminator(generated_images), valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            optimizer_D.zero_grad()
            
            # Real images loss
            real_loss = adversarial_loss(discriminator(real_images), valid)
            
            # Fake images loss
            fake_loss = adversarial_loss(discriminator(generated_images.detach()), fake)
            
            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            # Update progress
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}'
            })
        
        # Print epoch results
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        
        print(f"Epoch {epoch+1}/{epochs} - G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}")
        
        # Save models every 10 epochs
        if (epoch + 1) % 10 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(generator.state_dict(), f"models/generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"models/discriminator_epoch_{epoch+1}.pth")
            print(f"Models saved at epoch {epoch+1}")
        
        # Generate sample every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("Generating samples...")
            generator.eval()
            with torch.no_grad():
                sample_fmri = fmri_batch[:4]  # Take first 4 samples
                sample_real = real_images[:4]
                sample_generated = generator(sample_fmri)
                
                # Convert to numpy for analysis
                real_np = sample_real.cpu().numpy()
                gen_np = sample_generated.cpu().numpy()
                
                print(f"Real images range: [{real_np.min():.3f}, {real_np.max():.3f}]")
                print(f"Generated images range: [{gen_np.min():.3f}, {gen_np.max():.3f}]")
            generator.train()
    
    # Final save
    torch.save(generator.state_dict(), "models/generator_final.pth")
    torch.save(discriminator.state_dict(), "models/discriminator_final.pth")
    
    print("\nTraining completed!")
    print("Models saved in 'models/' directory")
    
    return generator, discriminator

if __name__ == "__main__":
    # Train the GAN
    generator, discriminator = train_gan(epochs=30, batch_size=16, lr=0.0002)
    
    print("\nTraining finished successfully!")
    print("You can now use the trained models for brain decoding.")
