import time
import os
import numpy as np
import pandas as pd
import torch

# SETTINGS
ROWS = 5_000_000  # 5 Million rows
COLS = 20         # 20 Features
BINS = 100        # We want to map floats to 0-99 (Tokenization)

def create_dummy_data():
    if not os.path.exists("massive_data.parquet"):
        print(f"Generating {ROWS:,} rows of dummy data...")
        # We generate in chunks to avoid blowing up RAM before we start
        df = pd.DataFrame(np.random.randn(ROWS, COLS), columns=[f"col_{i}" for i in range(COLS)])
        df.to_parquet("massive_data.parquet")
        print("Data generated and saved.")

def cpu_etl_pandas():
    print("\n--- CPU RUN (Pandas) ---")
    start = time.time()
    
    # 1. Load
    df = pd.read_parquet("massive_data.parquet")
    
    # 2. Quantile Binning (The Bottleneck)
    # We map every value to a bucket (0-100) based on distribution
    for col in df.columns:
        # qcut sorts the data to find quantiles -> slow on CPU
        df[col] = pd.qcut(df[col], q=BINS, labels=False, duplicates='drop')
        
    end = time.time()
    print(f"CPU Processing Time: {end - start:.2f} seconds")
    return df

def gpu_etl_pytorch():
    print("\n--- GPU RUN (PyTorch Tensor-ETL) ---")
    start = time.time()
    
    # 1. Load (Pandas is still needed to read the file, sadly)
    # Optimization: In production, we'd use a C++ loader, but this is fine for now
    df = pd.read_parquet("massive_data.parquet")
    
    # 2. Move to GPU (The overhead)
    # Convert entire dataframe to a single float32 tensor
    tensor_data = torch.tensor(df.values, dtype=torch.float32, device='cuda')
    
    # 3. Quantile Binning on GPU
    # We need to process each column independently
    
    # Pre-calculate quantile thresholds (0%, 1%... 100%)
    # This linspace is the % steps
    steps = torch.linspace(0, 1, BINS + 1, device='cuda')
    
    # Output tensor to hold the "tokens" (integers)
    tokenized_data = torch.zeros_like(tensor_data, dtype=torch.int32)
    
    # Loop through columns (The GPU is so fast, this loop is negligible)
    for i in range(COLS):
        col_data = tensor_data[:, i]
        
        # A. Find the boundaries (quantiles)
        # standard deviation-based quantiles or exact quantiles
        boundaries = torch.quantile(col_data, steps)
        
        # B. Bucketize (Map float to bin index)
        # This is the "Search Sorted" algorithm, massively parallel on GPU
        tokenized_data[:, i] = torch.bucketize(col_data, boundaries) - 1
        
        # Clamp to ensure 0-99 range (handle edges)
        tokenized_data[:, i] = torch.clamp(tokenized_data[:, i], 0, BINS - 1)
        
    # 4. Sync to finish timing
    torch.cuda.synchronize()
    
    end = time.time()
    print(f"GPU Processing Time: {end - start:.2f} seconds")
    return tokenized_data

if __name__ == "__main__":
    create_dummy_data()

    print(torch.cuda.is_available()) 

    print(torch.cuda.get_device_name(0))
    
    # Run CPU
    cpu_etl_pandas()
    
    # Run GPU
    gpu_etl_pytorch()