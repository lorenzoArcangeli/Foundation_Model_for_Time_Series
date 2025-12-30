
## Why the Embedding Shapes are Different
### Chronos-T5 ([1, 513, 512]):
**Architecture**: Uses Bit/Value Tokenization. It treats the time series almost like a sentence of text.
**Resolution**: It maps each time step (or small group) to a token.
**Result**: High-resolution embeddings. An input of ~512 time steps results in ~512 tokens (513 includes a start-special token).

### Chronos-Bolt ([1, 39, 512/768]) & Chronos-2 ([1, 38, 768]):
**Architecture**: Uses Patching (influenced by Vision Transformers).
**Resolution**: It splits the time series into "patches" (chunks of e.g., 32 or 16 time steps). Each chunk is mapped to a single vector.
**Result**: Compressed embeddings. Your 600 time steps are divided by the patch size, resulting in a much shorter sequence length (~38-39).
**Model Dimensions**: The last dimension (512 vs 768) reflects the model size (Small vs Base).
Summary for your Multimodal Project:
