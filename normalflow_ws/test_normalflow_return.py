#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'normalflow'))

from normalflow.registration import normalflow
import numpy as np

# Create dummy data
H, W = 100, 100
N_ref = np.random.rand(H, W, 3).astype(np.float32)
C_ref = np.random.rand(H, W) > 0.5
H_ref = np.random.rand(H, W).astype(np.float32)
N_tar = np.random.rand(H, W, 3).astype(np.float32)
C_tar = np.random.rand(H, W) > 0.5
H_tar = np.random.rand(H, W).astype(np.float32)

try:
    result = normalflow(N_ref, C_ref, H_ref, N_tar, C_tar, H_tar)
    print(f"Function returned: {type(result)}")
    if isinstance(result, tuple):
        print(f"Number of return values: {len(result)}")
        for i, val in enumerate(result):
            print(f"  Value {i}: {type(val)}, shape: {val.shape if hasattr(val, 'shape') else 'N/A'}")
    else:
        print(f"Single return value: {type(result)}, shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
except Exception as e:
    print(f"Error: {e}") 