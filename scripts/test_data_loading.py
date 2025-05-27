"""
Simple test script for data loading without visualization
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from data_loader import FMRIDataLoader
import numpy as np

def test_data_loading():
    """Test data loading functionality"""
    
    print("=== Testing Brain Decoding Data Loading ===\n")
    
    try:
        # Initialize the fMRI data loader
        loader = FMRIDataLoader(data_dir="data")
        
        # Load the brain decoding data
        print("1. Loading fMRI and stimulus data...")
        fmri_data, stimulus_data = loader.load_brain_decoding_data("digit69_28x28.mat")
        
        # Display data information
        print("\n2. Data Information:")
        info = loader.get_data_info()
        for data_type, data_info in info.items():
            print(f"   {data_type.upper()} Data:")
            for key, value in data_info.items():
                if isinstance(value, (int, float)):
                    print(f"     {key}: {value:.4f}")
                else:
                    print(f"     {key}: {value}")
            print()
        
        # Create PyTorch DataLoaders for brain decoding training
        print("3. Creating PyTorch DataLoaders for brain decoding...")
        train_loader, val_loader = loader.create_brain_decoding_dataloader(
            batch_size=16,  # Smaller batch size for testing
            shuffle=True,
            fmri_preprocessing="standardize",
            stimulus_normalize=True,
            train_split=0.8
        )
        
        print(f"   Created train loader with {len(train_loader)} batches")
        print(f"   Created validation loader with {len(val_loader)} batches")
        
        # Test loading a few batches
        print("\n4. Testing batch loading...")
        for i, (fmri_batch, stimulus_batch) in enumerate(train_loader):
            print(f"   Batch {i+1}:")
            print(f"     fMRI shape: {fmri_batch.shape}, dtype: {fmri_batch.dtype}")
            print(f"     Stimulus shape: {stimulus_batch.shape}, dtype: {stimulus_batch.dtype}")
            print(f"     fMRI range: [{fmri_batch.min():.3f}, {fmri_batch.max():.3f}]")
            print(f"     Stimulus range: [{stimulus_batch.min():.3f}, {stimulus_batch.max():.3f}]")
            
            if i >= 2:  # Only show first 3 batches
                break
        
        print("\n=== Data loading test completed successfully! ===")
        
        # Return dimensions for training
        sample_fmri, sample_stimulus = next(iter(train_loader))
        return {
            'fmri_dim': sample_fmri.shape[1],
            'img_channels': sample_stimulus.shape[1],
            'img_size': sample_stimulus.shape[2],
            'train_batches': len(train_loader),
            'val_batches': len(val_loader)
        }
        
    except Exception as e:
        print(f"Error during data loading test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_data_loading()
    if result:
        print(f"\nData dimensions for GAN training:")
        print(f"  fMRI dimension: {result['fmri_dim']}")
        print(f"  Image channels: {result['img_channels']}")
        print(f"  Image size: {result['img_size']}x{result['img_size']}")
        print(f"  Training batches: {result['train_batches']}")
        print(f"  Validation batches: {result['val_batches']}")
