import time
import os
import numpy as np
import pandas as pd
from numba import cuda

# SETTINGS: Let's double the data to make the GPU sweat
# (If you have <16GB RAM, keep this at 5_000_000)
ROWS = 10_000_000 
COLS = 20
BINS = 100

def create_dummy_data():
    if not os.path.exists("massive_data.parquet"):
        print(f"Generating {ROWS:,} rows of dummy data...")
        df = pd.DataFrame(np.random.randn(ROWS, COLS), columns=[f"col_{i}" for i in range(COLS)])
        df.to_parquet("massive_data.parquet")
        print("Data generated and saved.")

def cpu_etl_pandas():
    print("\n--- CPU RUN (Pandas) ---")
    
    # 1. Load (Disk IO)
    t0 = time.time()
    df = pd.read_parquet("massive_data.parquet")
    t1 = time.time()
    
    # 2. Compute (Quantile Binning)
    print("CPU Computing...")
    for col in df.columns:
        df[col] = pd.qcut(df[col], q=BINS, labels=False, duplicates='drop')
    t2 = time.time()
    
    print(f"  [Disk Load]: {t1 - t0:.4f} s")
    print(f"  [Compute]  : {t2 - t1:.4f} s")
    print(f"  TOTAL      : {t2 - t0:.4f} s")
    return t2 - t1 # Return only compute time for comparison

@cuda.jit
def bucketize_kernel(data, thresholds, output, rows, cols, bins):
    r = cuda.grid(1)
    if r < rows:
        for c in range(cols):
            val = data[r, c]
            bin_idx = bins - 1 
            for b in range(bins):
                if val < thresholds[c, b]:
                    bin_idx = b
                    break
            output[r, c] = bin_idx

def gpu_etl_numba():
    print("\n--- GPU RUN (Numba) ---")
    
    # 1. Load (Disk IO - purely CPU)
    t0 = time.time()
    df = pd.read_parquet("massive_data.parquet")
    # Convert to contiguous array (CPU work)
    host_data = np.ascontiguousarray(df.values, dtype=np.float32)
    # Calculate thresholds (CPU work)
    percentiles = np.linspace(0, 100, BINS+1)[1:]
    host_thresholds = np.percentile(host_data, percentiles, axis=0).T.astype(np.float32)
    t1 = time.time()
    
    # 2. Transfer (PCIe Bus)
    d_data = cuda.to_device(host_data)
    d_thresholds = cuda.to_device(host_thresholds)
    d_output = cuda.device_array((ROWS, COLS), dtype=np.int32)
    t2 = time.time()
    
    # 3. Compute (The Kernel)
    threads_per_block = 256
    blocks_per_grid = (ROWS + (threads_per_block - 1)) // threads_per_block
    
    print("GPU Computing...")
    # Synchronize before start to ensure clear timing
    cuda.synchronize()
    t_compute_start = time.time()
    
    bucketize_kernel[blocks_per_grid, threads_per_block](
        d_data, d_thresholds, d_output, ROWS, COLS, BINS
    )
    
    cuda.synchronize() # Wait for finish
    t_compute_end = time.time()
    
    print(f"  [Disk/Prep]: {t1 - t0:.4f} s")
    print(f"  [Transfer] : {t2 - t1:.4f} s")
    print(f"  [Compute]  : {t_compute_end - t_compute_start:.4f} s")
    print(f"  TOTAL      : {t_compute_end - t0:.4f} s")
    
    return t_compute_end - t_compute_start

if __name__ == "__main__":
    create_dummy_data()
    
    compute_cpu = cpu_etl_pandas()
    compute_gpu = gpu_etl_numba()
    
    print("\n" + "="*40)
    print(f"COMPUTE SPEEDUP: {compute_cpu / compute_gpu:.1f}x FASTER")
    print("="*40)