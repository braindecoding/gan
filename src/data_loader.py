import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Optional, Union, Dict
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainDecodingDataset(Dataset):
    """Custom Dataset class for Brain Decoding GAN training"""

    def __init__(self, fmri_data: np.ndarray, stimulus_data: np.ndarray,
                 fmri_transform=None, stimulus_transform=None):
        """
        Args:
            fmri_data: numpy array of fMRI data (N, features) or (N, voxels)
            stimulus_data: numpy array of stimulus images (N, H, W) or (N, H, W, C)
            fmri_transform: optional transform for fMRI data
            stimulus_transform: optional transform for stimulus images
        """
        assert len(fmri_data) == len(stimulus_data), "fMRI and stimulus data must have same length"

        self.fmri_data = fmri_data.astype(np.float32)
        self.stimulus_data = stimulus_data.astype(np.float32)
        self.fmri_transform = fmri_transform
        self.stimulus_transform = stimulus_transform

        # Ensure stimulus data is in the right format for PyTorch
        if len(self.stimulus_data.shape) == 3:
            # Add channel dimension if missing (N, H, W) -> (N, 1, H, W)
            self.stimulus_data = np.expand_dims(self.stimulus_data, axis=1)
        elif len(self.stimulus_data.shape) == 4 and self.stimulus_data.shape[-1] in [1, 3]:
            # Convert (N, H, W, C) to (N, C, H, W)
            self.stimulus_data = np.transpose(self.stimulus_data, (0, 3, 1, 2))

    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        fmri_sample = self.fmri_data[idx]
        stimulus_sample = self.stimulus_data[idx]

        if self.fmri_transform:
            fmri_sample = self.fmri_transform(fmri_sample)

        if self.stimulus_transform:
            stimulus_sample = self.stimulus_transform(stimulus_sample)

        return fmri_sample, stimulus_sample

class GANDataset(Dataset):
    """Custom Dataset class for simple GAN training (backward compatibility)"""

    def __init__(self, data: np.ndarray, transform=None):
        """
        Args:
            data: numpy array of shape (N, H, W) or (N, H, W, C)
            transform: optional transform to be applied on a sample
        """
        self.data = data
        self.transform = transform

        # Ensure data is in the right format
        if len(self.data.shape) == 3:
            # Add channel dimension if missing (N, H, W) -> (N, 1, H, W)
            self.data = np.expand_dims(self.data, axis=1)
        elif len(self.data.shape) == 4 and self.data.shape[-1] in [1, 3]:
            # Convert (N, H, W, C) to (N, C, H, W)
            self.data = np.transpose(self.data, (0, 3, 1, 2))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

class FMRIDataLoader:
    """Data loader for Brain Decoding GAN training"""

    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: directory containing data files
        """
        self.data_dir = data_dir
        self.fmri_data = None
        self.stimulus_data = None
        self.fmri_scaler = None
        self.stimulus_scaler = None

    def load_mat_file(self, filename: str) -> dict:
        """
        Load MATLAB .mat file

        Args:
            filename: name of the .mat file

        Returns:
            Dictionary containing the loaded data
        """
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found")

        logger.info(f"Loading data from {filepath}")
        data = loadmat(filepath)

        # Remove MATLAB metadata
        data = {k: v for k, v in data.items() if not k.startswith('__')}

        return data

    def load_brain_decoding_data(self, filename: str = "digit69_28x28.mat") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load fMRI and stimulus data from .mat file for brain decoding

        Args:
            filename: name of the brain data file

        Returns:
            Tuple of (fmri_data, stimulus_data)
        """
        mat_data = self.load_mat_file(filename)

        # Print available keys to help identify data structure
        logger.info(f"Available keys in {filename}: {list(mat_data.keys())}")

        # Try to find fMRI and stimulus data in common variable names
        fmri_keys = ['fmri', 'fMRI', 'fmriTrn', 'fmriTest', 'brain_data', 'neural_data', 'X', 'features', 'voxels']
        stimulus_keys = ['stimulus', 'stimuli', 'stimTrn', 'stimTest', 'images', 'targets', 'y', 'labels', 'digits']

        fmri_data = None
        stimulus_data = None

        # Find fMRI data
        for key in fmri_keys:
            if key in mat_data:
                fmri_data = mat_data[key]
                logger.info(f"Found fMRI data in key: '{key}'")
                break

        # Find stimulus data
        for key in stimulus_keys:
            if key in mat_data:
                stimulus_data = mat_data[key]
                logger.info(f"Found stimulus data in key: '{key}'")
                break

        # Special handling for train/test split data structure
        if fmri_data is None and stimulus_data is None:
            # Check if we have separate train/test files
            if 'fmriTrn' in mat_data and 'stimTrn' in mat_data:
                logger.info("Found separate train/test data structure")

                # Combine training and test data
                fmri_trn = mat_data['fmriTrn']
                stim_trn = mat_data['stimTrn']

                if 'fmriTest' in mat_data and 'stimTest' in mat_data:
                    fmri_test = mat_data['fmriTest']
                    stim_test = mat_data['stimTest']

                    # Combine train and test
                    fmri_data = np.vstack([fmri_trn, fmri_test])
                    stimulus_data = np.vstack([stim_trn, stim_test])

                    logger.info(f"Combined train ({fmri_trn.shape[0]}) and test ({fmri_test.shape[0]}) data")
                else:
                    # Use only training data
                    fmri_data = fmri_trn
                    stimulus_data = stim_trn
                    logger.info("Using only training data")

        # If still not found, try to infer from data shapes
        if fmri_data is None or stimulus_data is None:
            logger.info("Attempting to infer data types from shapes...")

            # Sort keys by array size
            array_keys = [(k, v) for k, v in mat_data.items() if isinstance(v, np.ndarray)]
            array_keys.sort(key=lambda x: x[1].size, reverse=True)

            for key, data in array_keys:
                logger.info(f"Key '{key}': shape {data.shape}, size {data.size}")

                # Heuristic: fMRI data usually has more features/voxels (2D: samples x features)
                # Stimulus data usually has spatial dimensions (3D or 4D: samples x height x width [x channels])
                if len(data.shape) == 2 and fmri_data is None:
                    fmri_data = data
                    logger.info(f"Inferred fMRI data from key: '{key}' (2D array)")
                elif len(data.shape) >= 3 and stimulus_data is None:
                    stimulus_data = data
                    logger.info(f"Inferred stimulus data from key: '{key}' (3D+ array)")

        if fmri_data is None:
            raise ValueError("Could not find fMRI data in the .mat file")
        if stimulus_data is None:
            raise ValueError("Could not find stimulus data in the .mat file")

        # Ensure data types and shapes are correct
        fmri_data = fmri_data.astype(np.float32)
        stimulus_data = stimulus_data.astype(np.float32)

        # Handle flattened stimulus data (e.g., 784 -> 28x28)
        if len(stimulus_data.shape) == 2:
            # Check if it's flattened image data
            n_samples, n_features = stimulus_data.shape

            # Common image sizes when flattened
            possible_sizes = {
                784: (28, 28),    # 28x28 = 784
                1024: (32, 32),   # 32x32 = 1024
                4096: (64, 64),   # 64x64 = 4096
                256: (16, 16),    # 16x16 = 256
            }

            if n_features in possible_sizes:
                h, w = possible_sizes[n_features]
                stimulus_data = stimulus_data.reshape(n_samples, h, w)
                logger.info(f"Reshaped flattened stimulus data from ({n_samples}, {n_features}) to ({n_samples}, {h}, {w})")
            else:
                logger.warning(f"Stimulus data has {n_features} features, cannot determine image dimensions")

        # Normalize stimulus data to [0, 1] if needed
        if stimulus_data.max() > 1.0:
            stimulus_data = stimulus_data / 255.0

        logger.info(f"Loaded fMRI data shape: {fmri_data.shape}")
        logger.info(f"Loaded stimulus data shape: {stimulus_data.shape}")
        logger.info(f"fMRI data range: [{fmri_data.min():.3f}, {fmri_data.max():.3f}]")
        logger.info(f"Stimulus data range: [{stimulus_data.min():.3f}, {stimulus_data.max():.3f}]")

        self.fmri_data = fmri_data
        self.stimulus_data = stimulus_data

        return fmri_data, stimulus_data

    def load_digit_data(self, filename: str = "digit69_28x28.mat") -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load digit data from .mat file (backward compatibility)

        Args:
            filename: name of the digit data file

        Returns:
            Tuple of (data, labels) where labels might be None
        """
        mat_data = self.load_mat_file(filename)

        # Try to find data and labels in common variable names
        data_keys = ['data', 'X', 'images', 'digits']
        label_keys = ['labels', 'y', 'targets', 'classes']

        data = None
        labels = None

        # Find data
        for key in data_keys:
            if key in mat_data:
                data = mat_data[key]
                break

        # If no common key found, use the largest array
        if data is None:
            largest_key = max(mat_data.keys(), key=lambda k: mat_data[k].size)
            data = mat_data[largest_key]
            logger.info(f"Using '{largest_key}' as data array")

        # Find labels
        for key in label_keys:
            if key in mat_data:
                labels = mat_data[key]
                break

        # Ensure data is float32 and normalized
        data = data.astype(np.float32)

        # Normalize to [0, 1] if not already
        if data.max() > 1.0:
            data = data / 255.0

        logger.info(f"Loaded data shape: {data.shape}")
        if labels is not None:
            logger.info(f"Loaded labels shape: {labels.shape}")

        self.data = data
        self.labels = labels

        return data, labels

    def preprocess_fmri_data(self, method: str = "standardize") -> np.ndarray:
        """
        Preprocess fMRI data

        Args:
            method: preprocessing method ("standardize", "normalize", or "none")

        Returns:
            Preprocessed fMRI data
        """
        if self.fmri_data is None:
            raise ValueError("No fMRI data loaded. Call load_brain_decoding_data() first.")

        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        if method == "standardize":
            if self.fmri_scaler is None:
                self.fmri_scaler = StandardScaler()
                processed_data = self.fmri_scaler.fit_transform(self.fmri_data)
            else:
                processed_data = self.fmri_scaler.transform(self.fmri_data)
            logger.info("fMRI data standardized (mean=0, std=1)")

        elif method == "normalize":
            if self.fmri_scaler is None:
                self.fmri_scaler = MinMaxScaler(feature_range=(-1, 1))
                processed_data = self.fmri_scaler.fit_transform(self.fmri_data)
            else:
                processed_data = self.fmri_scaler.transform(self.fmri_data)
            logger.info("fMRI data normalized to [-1, 1]")

        elif method == "none":
            processed_data = self.fmri_data.copy()
            logger.info("No preprocessing applied to fMRI data")

        else:
            raise ValueError("Method must be 'standardize', 'normalize', or 'none'")

        return processed_data.astype(np.float32)

    def create_brain_decoding_dataloader(self,
                                       batch_size: int = 32,
                                       shuffle: bool = True,
                                       fmri_preprocessing: str = "standardize",
                                       stimulus_normalize: bool = True,
                                       train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for brain decoding training

        Args:
            batch_size: batch size for training
            shuffle: whether to shuffle data
            fmri_preprocessing: preprocessing method for fMRI data
            stimulus_normalize: whether to normalize stimulus to [-1, 1]
            train_split: fraction of data to use for training

        Returns:
            Tuple of (train_loader, val_loader)
        """
        if self.fmri_data is None or self.stimulus_data is None:
            raise ValueError("No data loaded. Call load_brain_decoding_data() first.")

        # Preprocess fMRI data
        fmri_processed = self.preprocess_fmri_data(fmri_preprocessing)

        # Prepare stimulus data
        stimulus_processed = self.stimulus_data.copy()

        # Create transforms for stimulus data
        stimulus_transforms = []
        if stimulus_normalize:
            # Normalize to [-1, 1] for GAN training
            stimulus_transforms.append(transforms.Normalize([0.5], [0.5]))

        stimulus_transform = transforms.Compose(stimulus_transforms) if stimulus_transforms else None

        # Split data into train and validation
        n_samples = len(fmri_processed)
        n_train = int(n_samples * train_split)

        if shuffle:
            indices = np.random.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Create datasets
        train_dataset = BrainDecodingDataset(
            fmri_processed[train_indices],
            stimulus_processed[train_indices],
            stimulus_transform=stimulus_transform
        )

        val_dataset = BrainDecodingDataset(
            fmri_processed[val_indices],
            stimulus_processed[val_indices],
            stimulus_transform=stimulus_transform
        )

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        logger.info(f"Created train loader with {len(train_loader)} batches ({len(train_dataset)} samples)")
        logger.info(f"Created validation loader with {len(val_loader)} batches ({len(val_dataset)} samples)")

        return train_loader, val_loader

    def get_data_info(self) -> dict:
        """Get information about loaded data"""
        info = {}

        if self.fmri_data is not None:
            info["fmri"] = {
                "shape": self.fmri_data.shape,
                "dtype": self.fmri_data.dtype,
                "min_value": self.fmri_data.min(),
                "max_value": self.fmri_data.max(),
                "mean": self.fmri_data.mean(),
                "std": self.fmri_data.std()
            }

        if self.stimulus_data is not None:
            info["stimulus"] = {
                "shape": self.stimulus_data.shape,
                "dtype": self.stimulus_data.dtype,
                "min_value": self.stimulus_data.min(),
                "max_value": self.stimulus_data.max(),
                "mean": self.stimulus_data.mean(),
                "std": self.stimulus_data.std()
            }

        if not info:
            info = {"error": "No data loaded"}

        return info

    def create_dataloader(self,
                         batch_size: int = 64,
                         shuffle: bool = True,
                         normalize: bool = True,
                         augment: bool = False) -> DataLoader:
        """
        Create PyTorch DataLoader for training

        Args:
            batch_size: batch size for training
            shuffle: whether to shuffle data
            normalize: whether to normalize data to [-1, 1]
            augment: whether to apply data augmentation

        Returns:
            PyTorch DataLoader
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_digit_data() first.")

        # Create transforms
        transform_list = []

        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])

        if normalize:
            # Normalize to [-1, 1] for GAN training
            transform_list.append(transforms.Normalize([0.5], [0.5]))

        transform = transforms.Compose(transform_list) if transform_list else None

        # Create dataset
        dataset = GANDataset(self.data, transform=transform)

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )

        return dataloader

    def visualize_samples(self, num_samples: int = 16, figsize: Tuple[int, int] = (10, 10)):
        """
        Visualize random samples from the dataset

        Args:
            num_samples: number of samples to visualize
            figsize: figure size for matplotlib
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_digit_data() first.")

        # Select random samples
        indices = np.random.choice(len(self.data), num_samples, replace=False)
        samples = self.data[indices]

        # Calculate grid size
        grid_size = int(np.ceil(np.sqrt(num_samples)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        axes = axes.flatten()

        for i, sample in enumerate(samples):
            if len(sample.shape) == 3:
                # Remove channel dimension for visualization
                sample = sample.squeeze()

            axes[i].imshow(sample, cmap='gray')
            axes[i].axis('off')

            if self.labels is not None:
                axes[i].set_title(f'Label: {self.labels[indices[i]]}')

        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_brain_decoding_pairs(self, num_pairs: int = 8, figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize fMRI-stimulus pairs for brain decoding

        Args:
            num_pairs: number of pairs to visualize
            figsize: figure size for matplotlib
        """
        if self.fmri_data is None or self.stimulus_data is None:
            raise ValueError("No data loaded. Call load_brain_decoding_data() first.")

        # Select random samples
        indices = np.random.choice(len(self.fmri_data), num_pairs, replace=False)

        fig, axes = plt.subplots(2, num_pairs, figsize=figsize)

        for i, idx in enumerate(indices):
            # Plot fMRI data (as a heatmap or line plot)
            fmri_sample = self.fmri_data[idx]
            stimulus_sample = self.stimulus_data[idx]

            # Remove channel dimension for stimulus visualization if present
            if len(stimulus_sample.shape) == 3:
                stimulus_sample = stimulus_sample.squeeze()

            # Plot fMRI data
            if len(fmri_sample.shape) == 1:
                # 1D fMRI data - plot as line
                axes[0, i].plot(fmri_sample)
                axes[0, i].set_title(f'fMRI {idx}\n({len(fmri_sample)} features)')
            else:
                # 2D fMRI data - plot as heatmap
                im = axes[0, i].imshow(fmri_sample, cmap='viridis', aspect='auto')
                axes[0, i].set_title(f'fMRI {idx}\n{fmri_sample.shape}')
                plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

            axes[0, i].set_xlabel('Features/Voxels')

            # Plot stimulus image
            axes[1, i].imshow(stimulus_sample, cmap='gray')
            axes[1, i].set_title(f'Stimulus {idx}\n{stimulus_sample.shape}')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

# Backward compatibility class
class DataLoader(FMRIDataLoader):
    """Backward compatibility wrapper"""

    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir)
        self.data = None
        self.labels = None

# Example usage
if __name__ == "__main__":
    # Initialize data loader
    loader = DataLoader()

    # Load digit data
    data, labels = loader.load_digit_data()

    # Print data information
    info = loader.get_data_info()
    print("Data Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Create PyTorch DataLoader
    train_loader = loader.create_dataloader(batch_size=32, shuffle=True)

    print(f"\nDataLoader created with {len(train_loader)} batches")

    # Test loading a batch
    for batch in train_loader:
        print(f"Batch shape: {batch.shape}")
        break

    # Visualize some samples
    loader.visualize_samples(num_samples=9, figsize=(8, 8))
