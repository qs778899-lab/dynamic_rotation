#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'normalflow'))

from normalflow.registration import normalflow
import numpy as np

# Create dummy data similar to what would be used in the tracking
H, W = 100, 100
N_ref = np.random.rand(H, W, 3).astype(np.float32)
C_ref = np.random.rand(H, W) > 0.5
H_ref = np.random.rand(H, W).astype(np.float32)
N_curr = np.random.rand(H, W, 3).astype(np.float32)
C_curr = np.random.rand(H, W) > 0.5
H_curr = np.random.rand(H, W).astype(np.float32)
prev_T_ref = np.eye(4)
ppmm = 0.0634

print("Testing normalflow function...")
print(f"N_ref shape: {N_ref.shape}")
print(f"C_ref shape: {C_ref.shape}")
print(f"H_ref shape: {H_ref.shape}")
print(f"N_curr shape: {N_curr.shape}")
print(f"C_curr shape: {C_curr.shape}")
print(f"H_curr shape: {H_curr.shape}")
print(f"prev_T_ref shape: {prev_T_ref.shape}")
print(f"ppmm: {ppmm}")

try:
    # Test the exact same call pattern as in the tracking code
    result = normalflow(
        N_ref,
        C_ref,
        H_ref,
        N_curr,
        C_curr,
        H_curr,
        prev_T_ref,
        ppmm,
        5000,
        verbose=True
    )
    
    print(f"Function returned: {type(result)}")
    if isinstance(result, tuple):
        print(f"Number of return values: {len(result)}")
        for i, val in enumerate(result):
            print(f"  Value {i}: {type(val)}")
            if hasattr(val, 'shape'):
                print(f"    Shape: {val.shape}")
            if hasattr(val, '__len__'):
                print(f"    Length: {len(val)}")
    else:
        print(f"Single return value: {type(result)}")
        if hasattr(result, 'shape'):
            print(f"  Shape: {result.shape}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 