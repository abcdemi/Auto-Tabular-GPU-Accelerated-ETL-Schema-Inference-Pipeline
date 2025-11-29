# Tensor-Native ETL: GPU-Accelerated Data Ingestion

**High-Throughput Tabular Tokenization using PyTorch Primitives on RTX 3080.**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.4-orange)
![Hardware](https://img.shields.io/badge/GPU-RTX%203080-76B900)
![Speedup](https://img.shields.io/badge/Speedup-28x-brightgreen)

## 🚀 The Scenario
In **Tabular Foundation Model** pipelines (like TabPFN or TabTransformer), raw continuous data must be "Tokenized" (Quantile Binned) before entering the Neural Network.
*   **The Bottleneck:** Standard CPU ETL (Pandas `qcut`) is single-threaded and relies on slow sorting algorithms ($O(N \log N)$). For a dataset of **10 Million rows**, this creates an ~8-second bottleneck per batch.
*   **The Scale:** 200 Million scalar operations (10M rows $\times$ 20 columns).

## ⚡ The Solution
A **Tensor-Native ETL pipeline** that treats data ingestion as a massive matrix operation.
Instead of using Pandas for logic or writing raw kernels from scratch, we load raw memory into VRAM and utilize **PyTorch's optimized GPU primitives**:
1.  `torch.quantile`: Parallelized sorting and threshold calculation.
2.  `torch.bucketize`: Parallelized **Binary Search** to map floats to token IDs.

## 📊 Benchmarks

**Hardware:** NVIDIA GeForce RTX 3080 (10GB) vs. AMD Ryzen 5 CPU.
**Task:** Quantile Binning (100 bins) on 10,000,000 rows.

| Metric | CPU (Pandas) | GPU (PyTorch Native) | Speedup |
| :--- | :--- | :--- | :--- |
| **Compute Latency** | 7.68 s | **0.27 s** | **28.0x** 🚀 |
| **Data Transfer (PCIe)**| N/A | 0.52 s | Negligible |
| **Total Pipeline Time** | 8.55 s | 1.36 s | **6.3x** |

*> **Engineering Insight:** The pure compute speedup is 28x. Even when accounting for the overhead of moving 1.6GB of data over the PCIe bus (Transfer time), the GPU approach is still 6x faster end-to-end.*

## 🛠️ Code Deep Dive

The pipeline leverages PyTorch as a general-purpose numerical accelerator, not just for Deep Learning. This allows for seamless "Zero-Copy" handoff to the model trainer.

```python
# The "Magic" Tokenization Step

# 1. Pre-calculate quantiles (0%, 1%... 100%)
# GPU-accelerated sorting/selection
boundaries = torch.quantile(col_data, steps)

# 2. Binary Search on GPU
# Maps millions of floats to their bin index instantly
# torch.bucketize uses Binary Search (O(log N)), unlike standard Linear Search
tokens = torch.bucketize(col_data, boundaries) - 1
```

## 🧠 Engineering Decision: PyTorch vs. Custom Kernels

During development, I implemented two GPU approaches to solve this problem.

1.  **Approach A: Custom Numba Kernel** (Linear Search)
    *   *Result:* 0.48s Compute Time (~19x speedup).
    *   *Pros:* No heavy dependencies, runs on bare CUDA driver.
    *   *Cons:* Linear search complexity $O(N)$ per token.
2.  **Approach B: PyTorch Native** (Binary Search)
    *   *Result:* **0.27s Compute Time (~28x speedup).**
    *   *Pros:* Utilizes NVIDIA's highly optimized C++ primitives; Binary search complexity $O(\log N)$.
    *   *Cons:* Requires PyTorch installation.

**Verdict:** I selected **PyTorch** for the final production pipeline as it provided a **1.7x performance gain** over the custom kernel and integrates natively with the downstream model training loop.

## 💻 Usage

### Prerequisites
*   Python 3.8+
*   PyTorch with CUDA support (tested on 2.5.1+cu124)
*   `pip install pandas pyarrow`

### Run Benchmark
```bash
python etl_benchmark_pytorch.py
```

## 📝 License
MIT License.

---

### Author
**abcdemi**

*Targeting ML Systems & Foundation Model Infrastructure.*