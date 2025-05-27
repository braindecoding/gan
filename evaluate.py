#!/usr/bin/env python3
"""
Main evaluation script for Brain Decoding GAN
"""

import sys
import os
sys.path.append('src')
sys.path.append('scripts')

def main():
    print("Starting Brain Decoding GAN Evaluation...")
    
    try:
        from evaluation import comprehensive_evaluation
        results = comprehensive_evaluation()
        print("Comprehensive evaluation completed!")
        
        # Generate plots
        from plot_results import main as plot_main
        plot_main()
        print("Plots generated!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please check that all files are in the correct directories.")
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
