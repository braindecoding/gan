"""
Example usage of the data loader for Brain Decoding GAN training
"""

import torch
import matplotlib.pyplot as plt
from data_loader import FMRIDataLoader, DataLoader

def brain_decoding_example():
    """Example for brain decoding with fMRI data"""

    print("=== Brain Decoding GAN Data Loader Example ===\n")

    # Initialize the fMRI data loader
    loader = FMRIDataLoader(data_dir="data")

    try:
        # Load the brain decoding data
        print("1. Loading fMRI and stimulus data...")
        fmri_data, stimulus_data = loader.load_brain_decoding_data("digit69_28x28.mat")

        # Display data information
        print("\n2. Data Information:")
        info = loader.get_data_info()
        for data_type, data_info in info.items():
            print(f"   {data_type.upper()} Data:")
            for key, value in data_info.items():
                print(f"     {key}: {value}")
            print()

        # Create PyTorch DataLoaders for brain decoding training
        print("3. Creating PyTorch DataLoaders for brain decoding...")
        train_loader, val_loader = loader.create_brain_decoding_dataloader(
            batch_size=32,
            shuffle=True,
            fmri_preprocessing="standardize",  # Standardize fMRI data
            stimulus_normalize=True,          # Normalize stimulus to [-1, 1]
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

        # Visualize fMRI-stimulus pairs
        print("\n5. Visualizing fMRI-stimulus pairs...")
        print("   Close the plot window to continue...")
        loader.visualize_brain_decoding_pairs(num_pairs=6, figsize=(18, 8))

        # Example: Brain decoding GAN training loop structure
        print("\n6. Example brain decoding GAN training loop structure:")
        print("   for epoch in range(num_epochs):")
        print("       for batch_idx, (fmri_data, real_images) in enumerate(train_loader):")
        print("           # fmri_data shape:", next(iter(train_loader))[0].shape)
        print("           # real_images shape:", next(iter(train_loader))[1].shape)
        print("           # ")
        print("           # Generator: fmri_data -> fake_images")
        print("           # fake_images = generator(fmri_data)")
        print("           # ")
        print("           # Discriminator: real_images vs fake_images")
        print("           # Your brain decoding GAN training code here")
        print("           pass")

        print("\n=== Brain decoding data loading completed successfully! ===")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the data file exists in the 'data' folder")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    """Main function demonstrating data loading"""

    print("=== Standard GAN Data Loader Example ===\n")

    # Initialize the data loader
    loader = DataLoader(data_dir="data")

    try:
        # Load the digit data
        print("1. Loading digit data...")
        data, labels = loader.load_digit_data("digit69_28x28.mat")

        # Display data information
        print("\n2. Data Information:")
        info = loader.get_data_info()
        for key, value in info.items():
            print(f"   {key}: {value}")

        # Create PyTorch DataLoader for training
        print("\n3. Creating PyTorch DataLoader...")
        train_loader = loader.create_dataloader(
            batch_size=64,
            shuffle=True,
            normalize=True,  # Normalize to [-1, 1] for GAN training
            augment=False    # Set to True for data augmentation
        )

        print(f"   Created DataLoader with {len(train_loader)} batches")
        print(f"   Each batch contains {train_loader.batch_size} samples")

        # Test loading a few batches
        print("\n4. Testing batch loading...")
        for i, batch in enumerate(train_loader):
            print(f"   Batch {i+1}: shape = {batch.shape}, dtype = {batch.dtype}")
            print(f"   Min value: {batch.min():.3f}, Max value: {batch.max():.3f}")

            if i >= 2:  # Only show first 3 batches
                break

        # Visualize some samples
        print("\n5. Visualizing samples...")
        print("   Close the plot window to continue...")
        loader.visualize_samples(num_samples=16, figsize=(10, 10))

        # Example: Iterate through all data for training
        print("\n6. Example training loop structure:")
        print("   for epoch in range(num_epochs):")
        print("       for batch_idx, real_data in enumerate(train_loader):")
        print("           # real_data shape:", next(iter(train_loader)).shape)
        print("           # Your GAN training code here")
        print("           pass")

        print("\n=== Data loading completed successfully! ===")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the data file exists in the 'data' folder")
    except Exception as e:
        print(f"Unexpected error: {e}")

def test_different_batch_sizes():
    """Test different batch sizes"""
    print("\n=== Testing Different Batch Sizes ===")

    loader = DataLoader()
    loader.load_digit_data()

    batch_sizes = [16, 32, 64, 128]

    for batch_size in batch_sizes:
        try:
            dataloader = loader.create_dataloader(batch_size=batch_size)
            sample_batch = next(iter(dataloader))
            print(f"Batch size {batch_size}: {sample_batch.shape}")
        except Exception as e:
            print(f"Batch size {batch_size}: Error - {e}")

if __name__ == "__main__":
    # Run brain decoding example (main use case)
    brain_decoding_example()

    print("\n" + "="*60 + "\n")

    # Run standard GAN example (backward compatibility)
    main()

    # Test different batch sizes
    test_different_batch_sizes()
