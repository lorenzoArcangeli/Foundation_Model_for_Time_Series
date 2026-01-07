# Research Report: Amazon Chronos 2 Foundation Model

**Released:** October 2025 (Official: Oct 20, 2025)  
**Type:** Time Series Foundation Model  
**Architecture:** Encoder-Only Transformer (T5 Encoder-based)  
**Parameters:** 
*   **Base:** 120 Million
*   **Small:** 28 Million

## 1. Executive Summary
Amazon Chronos 2 represents a paradigm shift from its predecessor (Chronos v1, T5-based) and the efficiency-focused Chronos-Bolt. While it retains the "Language of Time Series" philosophy, it abandons the encoder-decoder Seq2Seq architecture in favor of a highly efficient **Encoder-Only** design. It is purpose-built for "Universal Forecasting," capable of handling univariate, multivariate, and covariate-rich time series in a zero-shot manner without fine-tuning.

## 2. Deep Architecture Analysis

### A. Encoder-Only Transformer
Unlike Chronos v1 (which used a standard T5 Encoder-Decoder to generate tokens autoregressively), Chronos 2 utilizes a pure encoder architecture. This allows it to process the entire context window simultaneously and understand global dependencies more effectively.
*   **Context Length:** Up to 8192 tokens.
*   **Prediction Length:** Up to 1024 tokens.

### B. Input Processing Pipeline
1.  **Scaling:** Input series are normalized (e.g., mean-scaling) to handle diverse magnitudes.
2.  **Patching & Embedding:** 
    *   The time series is divided into non-overlapping "patches".
    *   These patches are processed by a **Residual Network (ResNet)** to project them into high-dimensional embeddings.
    *   This contrasts with simple tokenization, allowing the model to capture local temporal structures within the tokens themselves.
3.  **Tokenization:** The model still leverages the core idea of mapping continuous values to a discrete vocabulary (quantization), enabling the Transformer to treat time series as a "language".

### C. The Core Innovation: Group Attention
To solve the limitation of standard Transformers in handling multivariate dependencies, Chronos 2 introduces a specialized **Group Attention** mechanism. The model alternates between two types of attention layers:
1.  **Time Attention:** Focuses on temporal dependencies *within* a single time series (Learning "What happens next?" based on history).
2.  **Group Attention:** Focuses on spatial/cross-series dependencies *across* multiple time series (Learning "How does Series A affect Series B?").

This mechanism allows native support for:
*   **Multivariate Forecasting:** Modeling correlations between co-evolving variables.
*   **Covariates:** Treating external factors (weather, holidays, promotions) as "groups" that interact with the target series via attention.
*   **Cross-Series Context:** Leveraging similar series in a batch to improve predictions via In-Context Learning (ICL).

## 3. Output Mechanism: Probabilistic Quantiles
While the internal representation uses tokenization, the output head is designed for rigorous probabilistic forecasting.
*   **Quantile Regression:** The model produces **multi-patch quantile outputs** (e.g., P10, P50, P90). 
*   This provides a distribution of possible future values rather than a single point estimate, which is critical for decision-making under uncertainty.
*   It predicts these quantiles for masked future patches, effectively filling in the "blanks" of the future.

## 4. Training Recipe

### Data Strategy
Chronos 2 was trained on a massive corpus combining:
*   **Real-World Data:** Large-scale public datasets.
*   **Synthetic Data:** A novel generation pipeline was used to create vast amounts of synthetic multivariate and covariate-rich data.
*   **Key Finding:** Research showed that the model trained on *purely synthetic data* performed nearly as well as the one trained on the mixed corpus, highlighting the power of diverse synthetic topology learning.

### Objective
The model is trained with a masked prediction objective (similar to BERT), where random patches of the future are masked, and the model must reconstruct the quantile distribution of those missing segments.

## 5. Performance & Benchmarks
*   **Speed:** Capable of forecasting **>300 time series per second** on a single A10G GPU.
*   **Accuracy:** Achieves State-of-the-Art (SOTA) zero-shot performance on benchmarks like **Chronos Benchmark II**, **fev-bench**, and **GIFT-Eval**.
*   **Comparison:** Outperforms Chronos v1 (>90% win rate), Chronos-Bolt, and Moirai.

## 6. Comparison: Chronos 1 vs. Chronos 2

| Feature | Chronos 1 (T5) | Chronos 2 |
| :--- | :--- | :--- |
| **Architecture** | Encoder-Decoder (Seq2Seq) | **Encoder-Only** |
| **Input Embedding** | Simple Tokenization | **ResNet-based Patch Embeddings** |
| **Multivariate** | No (Independent Univariate) | **Native (Group Attention)** |
| **Covariates** | Limited support | **Native Integration** |
| **Output** | Autoregressive Sampling | **Multi-patch Quantile Regression** |
| **Inference Speed** | Slow | **Fast (>300 series/sec)** |
| **Training Data** | Real + Synthetic (Gaussian) | **Real + Advanced Synthetic Multivariate** |
