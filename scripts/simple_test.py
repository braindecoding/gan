"""
Simple test without matplotlib
"""

import os
import sys
import numpy as np
import torch
from scipy.io import loadmat

def simple_test():
    """Simple test of data loading and GAN initialization"""
    
    print("=== Simple Brain Decoding Test ===\n")
    
    try:
        # 1. Load data directly
        print("1. Loading data...")
        data = loadmat("data/digit69_28x28.mat")
        
        # Combine train and test data
        fmri_data = np.vstack([data['fmriTrn'], data['fmriTest']])
        stim_data = np.vstack([data['stimTrn'], data['stimTest']])
        
        # Reshape stimulus data from flat to 28x28
        stim_data = stim_data.reshape(-1, 28, 28)
        
        # Normalize
        fmri_data = fmri_data.astype(np.float32)
        stim_data = stim_data.astype(np.float32) / 255.0  # Normalize to [0,1]
        
        print(f"   fMRI data: {fmri_data.shape}, range [{fmri_data.min():.3f}, {fmri_data.max():.3f}]")
        print(f"   Stimulus data: {stim_data.shape}, range [{stim_data.min():.3f}, {stim_data.max():.3f}]")
        
        # 2. Create simple dataset
        print("\n2. Creating dataset...")
        
        class SimpleBrainDataset:
            def __init__(self, fmri, stim):
                self.fmri = torch.FloatTensor(fmri)
                self.stim = torch.FloatTensor(stim).unsqueeze(1)  # Add channel dim
                
            def __len__(self):
                return len(self.fmri)
                
            def __getitem__(self, idx):
                return self.fmri[idx], self.stim[idx]
        
        dataset = SimpleBrainDataset(fmri_data, stim_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        
        print(f"   Dataset size: {len(dataset)}")
        print(f"   Number of batches: {len(dataloader)}")
        
        # 3. Test batch loading
        print("\n3. Testing batch loading...")
        for i, (fmri_batch, stim_batch) in enumerate(dataloader):
            print(f"   Batch {i+1}: fMRI {fmri_batch.shape}, Stimulus {stim_batch.shape}")
            if i >= 2:
                break
        
        # 4. Test simple GAN components
        print("\n4. Testing GAN components...")
        
        # Simple generator
        class SimpleGenerator(torch.nn.Module):
            def __init__(self, fmri_dim, img_size):
                super().__init__()
                self.fc = torch.nn.Sequential(
                    torch.nn.Linear(fmri_dim, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, img_size * img_size),
                    torch.nn.Tanh()
                )
                self.img_size = img_size
                
            def forward(self, x):
                out = self.fc(x)
                return out.view(-1, 1, self.img_size, self.img_size)
        
        # Simple discriminator
        class SimpleDiscriminator(torch.nn.Module):
            def __init__(self, img_size):
                super().__init__()
                self.fc = torch.nn.Sequential(
                    torch.nn.Linear(img_size * img_size, 512),
                    torch.nn.LeakyReLU(0.2),
                    torch.nn.Linear(512, 1),
                    torch.nn.Sigmoid()
                )
                
            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        # Initialize models
        fmri_dim = fmri_data.shape[1]
        img_size = 28
        
        generator = SimpleGenerator(fmri_dim, img_size)
        discriminator = SimpleDiscriminator(img_size)
        
        print(f"   Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
        print(f"   Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
        
        # 5. Test forward pass
        print("\n5. Testing forward pass...")
        sample_fmri, sample_stim = next(iter(dataloader))
        
        # Generator forward pass
        fake_images = generator(sample_fmri)
        print(f"   Generated images: {fake_images.shape}")
        
        # Discriminator forward pass
        real_pred = discriminator(sample_stim)
        fake_pred = discriminator(fake_images.detach())
        print(f"   Real predictions: {real_pred.shape}")
        print(f"   Fake predictions: {fake_pred.shape}")
        
        print("\n=== All tests passed! Ready for training ===")
        
        return {
            'fmri_dim': fmri_dim,
            'img_size': img_size,
            'dataset_size': len(dataset),
            'num_batches': len(dataloader)
        }
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = simple_test()
    if result:
        print(f"\nReady for training with:")
        print(f"  fMRI dimension: {result['fmri_dim']}")
        print(f"  Image size: {result['img_size']}x{result['img_size']}")
        print(f"  Dataset size: {result['dataset_size']} samples")
        print(f"  Batches per epoch: {result['num_batches']}")
