import time
import os
import numpy as np
import pandas as pd
from numba import cuda

# SETTINGS
ROWS = 5_000_000
COLS = 20
BINS = 100

def create_dummy_data():
    if not os.path.exists("massive_data.parquet"):
        print(f"Generating {ROWS:,} rows of dummy data...")
        df = pd.DataFrame(np.random.randn(ROWS, COLS), columns=[f"col_{i}" for i in range(COLS)])
        df.to_parquet("massive_data.parquet")
        print("Data generated and saved.")

# --- THE CPU VERSION ---
def cpu_etl_pandas():
    print("\n--- CPU RUN (Pandas) ---")
    start = time.time()
    
    df = pd.read_parquet("massive_data.parquet")
    
    # Quantile Binning (Slow on CPU)
    for col in df.columns:
        df[col] = pd.qcut(df[col], q=BINS, labels=False, duplicates='drop')
        
    end = time.time()
    print(f"CPU Processing Time: {end - start:.2f} seconds")
    return df

# --- THE GPU VERSION (NUMBA) ---
@cuda.jit
def bucketize_kernel(data, thresholds, output, rows, cols, bins):
    # Each thread handles one row
    r = cuda.grid(1)
    
    if r < rows:
        for c in range(cols):
            val = data[r, c]
            
            # Simple Linear Search to find the bin
            # (Checks if val < threshold[0], then threshold[1]...)
            bin_idx = bins - 1 
            for b in range(bins):
                if val < thresholds[c, b]:
                    bin_idx = b
                    break
            
            output[r, c] = bin_idx

def gpu_etl_numba():
    print("\n--- GPU RUN (Numba Custom Kernel) ---")
    start = time.time()
    
    # 1. Load Data
    df = pd.read_parquet("massive_data.parquet")
    
    # 2. Prepare Data for GPU (Float32 is faster)
    host_data = np.ascontiguousarray(df.values, dtype=np.float32)
    
    # Calculate thresholds on CPU (Numpy is fast enough for this)
    # This creates the "Bin Edges" for every column
    percentiles = np.linspace(0, 100, BINS+1)[1:]
    host_thresholds = np.percentile(host_data, percentiles, axis=0).T.astype(np.float32)
    
    # 3. Move to GPU
    d_data = cuda.to_device(host_data)
    d_thresholds = cuda.to_device(host_thresholds)
    d_output = cuda.device_array((ROWS, COLS), dtype=np.int32)
    
    # 4. Launch Kernel
    threads_per_block = 256
    blocks_per_grid = (ROWS + (threads_per_block - 1)) // threads_per_block
    
    bucketize_kernel[blocks_per_grid, threads_per_block](
        d_data, d_thresholds, d_output, ROWS, COLS, BINS
    )
    cuda.synchronize()
    
    end = time.time()
    print(f"GPU Processing Time: {end - start:.2f} seconds")

if __name__ == "__main__":
    create_dummy_data()
    cpu_etl_pandas()
    gpu_etl_numba()