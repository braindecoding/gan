#!/usr/bin/env python3
"""
Main training script for Brain Decoding GAN
"""

import sys
import os
sys.path.append('src')

def main():
    from train import train_brain_decoding_gan
    
    print("Starting Brain Decoding GAN Training...")
    
    # Default parameters
    train_brain_decoding_gan(
        epochs=50,
        batch_size=16,
        lr=0.0002
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
