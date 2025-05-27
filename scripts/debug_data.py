"""
Debug script to check data structure
"""

import scipy.io
import numpy as np

def debug_data():
    """Debug data loading step by step"""
    
    print("=== Debugging Data Structure ===\n")
    
    try:
        # Load raw .mat file
        print("1. Loading raw .mat file...")
        data = scipy.io.loadmat("data/digit69_28x28.mat")
        
        # Show all keys
        print("\n2. Available keys:")
        for key in data.keys():
            if not key.startswith('__'):
                value = data[key]
                if hasattr(value, 'shape'):
                    print(f"   {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"   {key}: {type(value)}")
        
        # Check specific keys
        print("\n3. Detailed analysis:")
        
        if 'fmriTrn' in data:
            fmri_trn = data['fmriTrn']
            print(f"   fmriTrn: {fmri_trn.shape}, range [{fmri_trn.min():.3f}, {fmri_trn.max():.3f}]")
        
        if 'stimTrn' in data:
            stim_trn = data['stimTrn']
            print(f"   stimTrn: {stim_trn.shape}, range [{stim_trn.min():.3f}, {stim_trn.max():.3f}]")
            
            # Check if it's 784 = 28x28
            if stim_trn.shape[1] == 784:
                print(f"   stimTrn appears to be flattened 28x28 images")
                
                # Try reshaping one sample
                sample = stim_trn[0].reshape(28, 28)
                print(f"   Reshaped sample: {sample.shape}, range [{sample.min():.3f}, {sample.max():.3f}]")
        
        if 'fmriTest' in data:
            fmri_test = data['fmriTest']
            print(f"   fmriTest: {fmri_test.shape}, range [{fmri_test.min():.3f}, {fmri_test.max():.3f}]")
        
        if 'stimTest' in data:
            stim_test = data['stimTest']
            print(f"   stimTest: {stim_test.shape}, range [{stim_test.min():.3f}, {stim_test.max():.3f}]")
        
        # Try combining data
        print("\n4. Combining train and test data...")
        if 'fmriTrn' in data and 'fmriTest' in data:
            combined_fmri = np.vstack([data['fmriTrn'], data['fmriTest']])
            print(f"   Combined fMRI: {combined_fmri.shape}")
        
        if 'stimTrn' in data and 'stimTest' in data:
            combined_stim = np.vstack([data['stimTrn'], data['stimTest']])
            print(f"   Combined stimulus: {combined_stim.shape}")
            
            # Reshape to images
            if combined_stim.shape[1] == 784:
                reshaped_stim = combined_stim.reshape(-1, 28, 28)
                print(f"   Reshaped stimulus: {reshaped_stim.shape}")
        
        print("\n=== Data structure analysis completed ===")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_data()
