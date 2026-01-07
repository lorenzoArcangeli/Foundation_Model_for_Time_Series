# Research Report: Amazon Chronos 2 Foundation Model

**Released:** October 2025  
**Type:** Time Series Foundation Model  
**Architecture:** 120M Parameter Encoder-Only Transformer

## Overview
Chronos 2 is the latest foundation model for time series forecasting from Amazon, succeeding the original T5-based Chronos (v1) and the distilled Chronos-Bolt. It represents a significant architectural shift from an encoder-decoder (Seq2Seq) framework to a more efficient **encoder-only architecture** specifically designed to handle universal forecasting tasks—univariate, multivariate, and covariate-informed—in a zero-shot manner.

## Core Architecture & Mechanism

### 1. Encoder-Only Design
Unlike Chronos v1 which used a T5 encoder-decoder, Chronos 2 uses a 120-million parameter encoder-only architecture. This design is generally more efficient for "understanding" and processing the full context of a time series window to generate predictions.

### 2. Group Attention Mechanism
The defining innovation of Chronos 2 is the **Group Attention** layer. The model alternates between two types of attention blocks:
- **Time Attention:** Aggregates information *within* a single time series (temporal dependencies).
- **Group Attention:** Aggregates information *across* multiple time series (spatial/cross-series dependencies).

This allows the model to effectively process:
- **Multivariate Data:** Modeling dependencies between different variables (e.g., price and demand).
- **Covariates:** Incorporating external factors (e.g., weather, holidays) as related "groups" of data rather than just concatenated features.
- **Cross-Series Learning:** Sharing information dynamically between related series in a dataset.

### 3. Input Processing: Tokenization & Quantization
Chronos 2 retains the core "Language of Time Series" philosophy from v1 but refines it:
- **Scaling:** Data is first normalized (typically mean-scaled) to handle varying magnitudes.
- **Quantization (TSBin):** Continuous values are mapped to a fixed vocabulary of discrete tokens (e.g., 4096 bins). This converts the numerical time series into a sequence of tokens that the Transformer can process like text.
- **Special Tokens:** Uses standard tokens like `PAD` (padding) and `EOS` (end of sequence) to manage variable lengths and structure.

## key Features & Capabilities

*   **Universal Zero-Shot Forecasting:** Capable of forecasting unseen datasets without fine-tuning, handling arbitrary mix of univariate and multivariate data.
*   **In-Context Learning (ICL):** Leveraging the encoder architecture, it adapts to new patterns and tasks purely from the context provided in the input window.
*   **Covariate Support:** Native handling of past-known and future-known covariates (like scheduled events), which was a limitation in simpler univariate models.
*   **Performance:** Benchmarks show it outperforms Chronos v1 (T5), Chronos-Bolt, and other foundation models (like Moirai) with a >90% win rate on head-to-head evaluations.

## Comparison to Chronos 1
| Feature | Chronos 1 (T5) | Chronos 2 |
| :--- | :--- | :--- |
| **Architecture** | Encoder-Decoder (T5 based) | **Encoder-Only** |
| **Multivariate** | Via independent univariate handling (mostly) | **Native via Group Attention** |
| **Covariates** | Limited / Concatenation | **Native Group integration** |
| **Efficiency** | Slower (Autoregressive decoding) | **Faster / More Efficient** |
| **Context** | Standard Attention | **Alternating Time/Group Attention** |

## Conclusion
Chronos 2 is a major evolutionary step, moving away from repurposed NLP architectures (T5) towards a custom-designed Transformer architecture for time series that specifically addresses the challenges of multivariate relationships and external regressors through its Group Attention mechanism.
