Here is the polished `README.md` for your Data Engineering project.

I have framed it to specifically highlight **High-Performance Computing (HPC)** skills, which is exactly what a startup building Foundation Models (like PriorLabs) looks for.

***

# Auto-Tabular: GPU-Accelerated ETL for Foundation Models

**High-throughput Quantile Binning (Tokenization) pipeline using Custom Numba CUDA Kernels.**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Numba](https://img.shields.io/badge/Numba-CUDA-green)
![Hardware](https://img.shields.io/badge/GPU-RTX%203080-76B900)
![Focus](https://img.shields.io/badge/Focus-ML%20Infrastructure-orange)

## 🚀 The Problem
Training **Tabular Foundation Models** (like TabPFN) requires ingesting terabytes of heterogeneous tabular data. A critical pre-processing step is **Tokenization**: converting continuous floating-point numbers into discrete integer bins (quantiles) so Transformers can process them.

*   **The Bottleneck:** Standard CPU approaches (Pandas `qcut` / Scikit-Learn `KBinsDiscretizer`) rely on sorting and heavy branching. When processing millions of rows across thousands of datasets, the CPU becomes the ingestion bottleneck, starving the GPU during training.
*   **The Scale:** 10 Million Rows $\times$ 20 Features = 200 Million operations.

## ⚡ The Solution
**Auto-Tabular** is a lightweight ETL engine that offloads the heavy quantization logic to the GPU.
Instead of using heavy Deep Learning frameworks (like PyTorch) which add gigabytes of dependency overhead, this project implements a **raw CUDA kernel via Numba**. This allows for "Bare Metal" performance on any NVIDIA GPU with minimal setup.

### Key Architecture
1.  **Hybrid Execution:**
    *   **CPU:** Handles IO and statistical inference (calculating percentiles/schema).
    *   **GPU:** Handles the massive parallel transformation (mapping floats to tokens).
2.  **Custom CUDA Kernel:**
    *   Implements a parallelized **Linear Search** to map values to bins.
    *   Minimizes memory bandwidth usage by processing data in contiguous float32 blocks.
    *   Zero-dependency deployment (requires only NVIDIA Drivers, no CUDA Toolkit).

## 📊 Performance Benchmarks

Tested on **NVIDIA GeForce RTX 3080** (10GB VRAM) processing a dataset of **10,000,000 Rows** (200 Million scalar values).

| Metric | CPU (Pandas Optimized) | GPU (Numba Kernel) | Speedup |
| :--- | :--- | :--- | :--- |
| **Compute Time** | 9.13 s | **0.48 s** | **18.9x** 🚀 |
| **Throughput** | ~21M items/sec | **~416M items/sec** | **~20x** |
| **Disk/Prep Overhead** | 0.51 s | 5.67 s* | N/A |

*> **Engineering Insight:** While the GPU compute is ~19x faster, the total wall time is currently bottlenecked by the CPU calculating percentile thresholds (Prep). A future optimization would move the sorting/percentile logic to the GPU using **Radix Sort** (via RAPIDS cuDF) to eliminate this final bottleneck.*

## 🛠️ Code Deep Dive

The core of the project is a JIT-compiled CUDA kernel that maps continuous values to bucket indices in parallel:

```python
@cuda.jit
def bucketize_kernel(data, thresholds, output, rows, cols, bins):
    # Global Thread ID (Row Index)
    r = cuda.grid(1)
    
    if r < rows:
        # Loop through columns for this specific row
        for c in range(cols):
            val = data[r, c]
            bin_idx = bins - 1 
            
            # Fast Linear Search against column-specific thresholds
            for b in range(bins):
                if val < thresholds[c, b]:
                    bin_idx = b
                    break
            
            output[r, c] = bin_idx
```

## 💻 Installation & Usage

### Prerequisites
*   Python 3.8+
*   NVIDIA GPU
*   `pip install numpy pandas numba pyarrow`

### Running the Benchmark
```bash
# Clone the repository
git clone https://github.com/yourusername/Auto-Tabular-GPU-ETL.git

# Run the profiled benchmark
python etl_benchmark_numba.py
```

## 🧠 Why Numba?
While PyTorch or TensorFlow could solve this, they require the full CUDA Toolkit and 2GB+ of libraries. **Numba** interacts directly with the CUDA Driver API. This makes `Auto-Tabular` extremely portable—it can run on any machine with NVIDIA drivers, making it ideal for lightweight data ingestion containers in a Kubernetes cluster.

## 📝 License
MIT License.

---

### Author
**[Your Name]**
*Targeting ML Systems & Foundation Model Infrastructure.*