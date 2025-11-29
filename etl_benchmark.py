import time
import os
import numpy as np
import pandas as pd
import torch

# --- CONFIGURATION ---
# 10 Million Rows (Approx 800MB - 1.5GB in RAM)
# RTX 3080 (10GB VRAM) handles this easily.
ROWS = 10_000_000 
COLS = 20
BINS = 100

def create_dummy_data():
    if not os.path.exists("massive_data.parquet"):
        print(f"Generating {ROWS:,} rows of dummy data...")
        # Generate on CPU (Numpy)
        df = pd.DataFrame(np.random.randn(ROWS, COLS), columns=[f"col_{i}" for i in range(COLS)])
        df.to_parquet("massive_data.parquet")
        print("Data generated and saved.")

def cpu_etl_pandas():
    print("\n--- CPU RUN (Pandas) ---")
    
    # 1. Disk IO
    t0 = time.time()
    df = pd.read_parquet("massive_data.parquet")
    t1 = time.time()
    
    # 2. Compute (Quantile Binning)
    print("CPU Computing (qcut)...")
    for col in df.columns:
        # pd.qcut sorts data under the hood -> Slow
        df[col] = pd.qcut(df[col], q=BINS, labels=False, duplicates='drop')
    t2 = time.time()
    
    print(f"  [Disk Load]: {t1 - t0:.4f} s")
    print(f"  [Compute]  : {t2 - t1:.4f} s")
    print(f"  TOTAL      : {t2 - t0:.4f} s")
    return t2 - t1

def gpu_etl_pytorch():
    print("\n--- GPU RUN (PyTorch Native) ---")
    
    # 1. Disk IO (CPU)
    t0 = time.time()
    df = pd.read_parquet("massive_data.parquet")
    t1 = time.time()
    
    # 2. Transfer to GPU
    # Convert Pandas (CPU) -> PyTorch Tensor (GPU)
    # This moves data over the PCIe bus
    tensor_data = torch.tensor(df.values, dtype=torch.float32, device='cuda')
    t2 = time.time()
    
    # 3. Compute (The Magic)
    print("GPU Computing (quantile + bucketize)...")
    torch.cuda.synchronize() # Clear queue
    t_compute_start = time.time()
    
    # Pre-calculate steps (0.0, 0.01, ... 1.0)
    # We use torch.float32 to match data
    steps = torch.linspace(0, 1, BINS + 1, device='cuda', dtype=torch.float32)
    
    # Output tensor for tokens (Integers)
    # Allocating memory beforehand is faster
    tokenized_data = torch.empty_like(tensor_data, dtype=torch.int32)
    
    # Loop columns (GPU loop overhead is negligible compared to math)
    for i in range(COLS):
        col_data = tensor_data[:, i]
        
        # A. Calculate Quantiles (The heavy sorting step)
        boundaries = torch.quantile(col_data, steps)
        
        # B. Map values to bins (Binary Search)
        # torch.bucketize is heavily optimized
        # We subtract 1 to get 0-indexed bins (0 to 99)
        tokens = torch.bucketize(col_data, boundaries) - 1
        
        # C. Clamp to ensure safety (handle edges)
        tokenized_data[:, i] = torch.clamp(tokens, 0, BINS - 1)
        
    torch.cuda.synchronize() # Wait for GPU to finish
    t_compute_end = time.time()
    
    print(f"  [Disk Load]: {t1 - t0:.4f} s")
    print(f"  [Transfer] : {t2 - t1:.4f} s")
    print(f"  [Compute]  : {t_compute_end - t_compute_start:.4f} s")
    print(f"  TOTAL      : {t_compute_end - t0:.4f} s")
    
    return t_compute_end - t_compute_start

if __name__ == "__main__":
    # Check GPU
    if not torch.cuda.is_available():
        print("Error: PyTorch cannot find GPU. Check cuda installation.")
        exit()
        
    create_dummy_data()
    
    time_cpu = cpu_etl_pandas()
    time_gpu = gpu_etl_pytorch()
    
    print("\n" + "="*40)
    print(f"COMPUTE SPEEDUP: {time_cpu / time_gpu:.1f}x FASTER")
    print("="*40)