# Time Complexity (TC) & Computational Efficiency Audit Report
**Project:** Inventory Demand Forecasting via Deep Learning (Phase 2)  
**Evaluated Architecture:** `NormalDemandMLP` (Multi-Layer Perceptron with Entity Embeddings)  
**Device Context:** Local Evaluation Environment  

---

## 1. Theoretical Complexity Analysis (Big-O Formulation)

### A. Time Complexity (Inference & Forward Pass)
For a single demand prediction:
1. **Entity Embedding Lookups:** For $K$ categorical features, embedding table lookup is $O(1)$ constant time per feature:
   $$\mathcal{O}_{\text{emb}} = \mathcal{O}(K)$$
2. **Dense Multi-Layer Perceptron Stack:** With $L$ dense layers having dimensions $d_0, d_1, \dots, d_L$:
   $$\mathcal{O}_{\text{dense}} = \sum_{l=1}^{L} \mathcal{O}(d_{l-1} \cdot d_l)$$
3. **Total Inference Time Complexity per SKU:**
   $$\mathcal{O}_{\text{inference}} = \mathcal{O}\left(K + \sum_{l=1}^{L} d_{l-1} \cdot d_l\right)$$
   For batch size $B$, the vectorized forward pass operates in:
   $$\mathcal{O}_{\text{batch}} = \mathcal{O}\left(B \cdot \sum_{l=1}^{L} d_{l-1} \cdot d_l\right)$$

### B. Time Complexity (Training Pass)
For $N$ training samples over $E$ epochs with backpropagation:
- Forward propagation: $\mathcal{O}(B \cdot \sum d_{l-1} d_l)$
- Gradient backpropagation (matrix-vector transposed multiplications): $\approx 2 \times \mathcal{O}(B \cdot \sum d_{l-1} d_l)$
- Total Training Time Complexity:
  $$\mathcal{O}_{\text{training}} = \mathcal{O}\left(E \cdot N \cdot \sum_{l=1}^{L} d_{l-1} \cdot d_l\right)$$

### C. Space Complexity (Memory & Parameter Footprint)
1. **Static Parameter Space:**
   $$\mathcal{S}_{\text{params}} = \mathcal{O}\left(\sum_{k=1}^{K} |C_k| \cdot d_{e,k} + \sum_{l=1}^{L} (d_{l-1} \cdot d_l + d_l)\right)$$
2. **Activation Memory during Training (Backpropagation Graph):**
   $$\mathcal{S}_{\text{activations}} = \mathcal{O}\left(B \cdot \sum_{l=1}^{L} d_l\right)$$
3. **Inference Memory (Peak Footprint):**
   Only active layer activations are retained:
   $$\mathcal{S}_{\text{inference}} = \mathcal{O}(B \cdot \max_{l} d_l)$$

---

## 2. Empirical Parameter & FLOP Profile

| Architectural Metric | Quantitative Value | Operational Interpretation |
| :--- | :--- | :--- |
| **Total Parameter Count** | **71,791** | Lightweight, highly compact model footprint |
| **Trainable Parameters** | **71,791** | 100% differentiable end-to-end |
| **Embedding Parameters** | **6,062** (8.4%) | Sparse categorical representation |
| **Dense MLP Parameters** | **65,729** (91.6%) | Core representation and regression layers |
| **FLOPs per Sample** | **128,576 FLOPs** | < 0.15 MFLOPs (Extremely low compute cost) |
| **Model Size in RAM** | **~280.43 KB** (float32) | Trivially cacheable in L2/L3 CPU cache |

---

## 3. Empirical Latency & Throughput Benchmark

Evaluated over 100 iterations per batch size:

| Batch Size ($B$) | Batch Latency (ms) | Latency per SKU (μs) | Throughput (SKUs / sec) |
| :---: | :---: | :---: | :---: |
| 1 | 0.067 ms | 66.82 μs | **14,964.7** |
| 8 | 0.119 ms | 14.88 μs | **67,200.1** |
| 16 | 0.132 ms | 8.23 μs | **121,501.7** |
| 32 | 0.141 ms | 4.41 μs | **226,602.8** |
| 64 | 0.175 ms | 2.73 μs | **366,331.0** |
| 128 | 0.208 ms | 1.62 μs | **616,751.0** |
| 256 | 0.279 ms | 1.09 μs | **916,102.9** |

---

## 4. Academic Evaluation Verdict (TC Check)

1. **Sub-millisecond Single-SKU Latency:** At batch size $B=1$, the model responds in under **0.07 ms**, qualifying it for real-time edge or point-of-sale deployment.
2. **High-Volume Throughput:** At batch size $B=256$, throughput achieves over **916,103 SKU forecasts per second**, meaning the entire Walmart chain (30,490 SKUs) can be forecasted in under **0.03 seconds**!
3. **No Memory Bottlenecks:** With a memory footprint under 1 MB, the normal neural network exhibits zero risk of Out-of-Memory (OOM) errors during both training and inference.
