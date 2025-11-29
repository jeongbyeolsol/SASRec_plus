# SASRec-Plus

SASRec-Plus is an enhanced implementation of the Self-Attentive Sequential Recommendation (SASRec) model.  
It integrates the original SASRec architecture with additional modules such as hybrid positional encoding, configurable embedding sharing, and improved masking logic for sequential user–item interaction modeling.

---

## 🚀 Key Features

- **Self-Attention-based Sequential Recommendation**
  - Transformer encoder-style architecture
  - Causal attention ensures future items are not visible

- **Hybrid Positional Encoding**
  - Learnable embedding positional encoding
  - Optional sinusoidal encoding (`fixed_pos_embd = 1 or 2`)
  - `concat` mode supports linear projection for fusion

- **Flexible Design**
  - Variable number of layers, hidden dimension, and dropout
  - Embedding weight sharing option (`share_embd=True`) for decoding

- **Masking**
  - **Padding Mask** removes irrelevant padded positions
  - **Causal Mask** prevents information leakage into the future

- **Pipeline Integration**
  - `SASRecPipeline` provides:
    - model training
    - evaluation
    - checkpoint saving/loading
    - NDCG / HitRate computation (with negative sampling, à la SASRec paper)

---

## 📁 Project Structure
```
sasrec_plus.py # SASRec-Plus model implementation
README.md # This documentation
MicroLens-100k_user_sequences.json # Dataset for model input
```



## 🔧 Usage Example

```python
from sasrec_plus import SASRecPipeline

pipeline = SASRecPipeline(
    num_items=...,
    max_len=50,
    d_model=64,
    n_layers=2,
    dropout=0.2,
)

pipeline.model_train(epoch=10)
pipeline.model_validate()
pipeline.model_test()

```


## 🧪 Dataset

This implementation uses the MicroLens-100k dataset.
Each user sequence should be padded (e.g., PAD_ID = 0) to form a (batch_size, max_len) tensor.

## 📦 Requirements
```
Python >= 3.8
PyTorch
tqdm
numpy
json
```

## ✨ Notes

The model supports both learnable and fixed positional encodings.

For evaluation, negative sampling (default: 100 negatives per user) follows the protocol in the SASRec paper.

Checkpointing keeps both the best-NDCG model and the last epoch model separately.

## Reference

Kang, Wang-Cheng, and Julian McAuley. "Self-attentive sequential recommendation." 2018 IEEE international conference on data mining (ICDM). IEEE, 2018.


