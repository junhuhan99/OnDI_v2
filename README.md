# OnDI v2 - Korean-English Bilingual Language Model

## Overview

OnDI v2 is a **983M parameter** decoder-only transformer language model pre-trained from scratch on 100GB of high-quality data. This is the **pre-trained base model** before SFT (Supervised Fine-Tuning).

## Model Details

| Specification | Value |
|---------------|-------|
| **Parameters** | 983,314,944 (~983M) |
| **Architecture** | Decoder-only Transformer (GPT-style) |
| **Vocabulary Size** | 50,257 |
| **Hidden Size** | 1,536 |
| **Layers** | 24 |
| **Attention Heads** | 16 |
| **Intermediate Size** | 6,144 |
| **Max Sequence Length** | 2,048 |
| **Positional Encoding** | Learned Positional Embeddings |

## Training Details

### Pre-training Data
- **Total Size**: 100GB high-quality text
- **Composition**:
  - 70% English text
  - 30% Code (Python, JavaScript, etc.)

### Training Configuration
- **Hardware**: 4x NVIDIA Tesla T4 (16GB each)
- **Framework**: PyTorch + DeepSpeed ZeRO-3
- **Training Steps**: 32,000
- **Batch Size**: 64 (effective)
- **Learning Rate**: 3e-4 with cosine decay
- **Warmup Steps**: 1,000
- **Precision**: FP16 mixed precision
- **Training Time**: ~10 hours

## Model Architecture

```python
OndiModel(
  embed_tokens: Embedding(50257, 1536)
  embed_positions: Embedding(2048, 1536)
  layers: ModuleList(
    (0-23): 24 x OndiBlock(
      ln1: LayerNorm(1536)
      attn: OndiAttention(
        q_proj: Linear(1536, 1536)
        k_proj: Linear(1536, 1536)
        v_proj: Linear(1536, 1536)
        o_proj: Linear(1536, 1536)
      )
      ln2: LayerNorm(1536)
      mlp: OndiMLP(
        gate_proj: Linear(1536, 6144)
        up_proj: Linear(1536, 6144)
        down_proj: Linear(6144, 1536)
      )
    )
  )
  norm: LayerNorm(1536)
  lm_head: Linear(1536, 50257)
)
```

## Usage

### Loading the Model

```python
import torch
import torch.nn as nn

# Model configuration
config = {
    "vocab_size": 50257,
    "max_position_embeddings": 2048,
    "hidden_size": 1536,
    "num_layers": 24,
    "num_heads": 16,
    "intermediate_size": 6144,
}

# Define model architecture (same as training)
class OndiMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.gelu(self.gate_proj(x)) * self.up_proj(x))

class OndiAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

class OndiBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config["hidden_size"])
        self.attn = OndiAttention(config["hidden_size"], config["num_heads"])
        self.ln2 = nn.LayerNorm(config["hidden_size"])
        self.mlp = OndiMLP(config["hidden_size"], config["intermediate_size"])

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class OndiModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.embed_positions = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.layers = nn.ModuleList([OndiBlock(config) for _ in range(config["num_layers"])])
        self.norm = nn.LayerNorm(config["hidden_size"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(self, input_ids):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

# Load model
model = OndiModel(config)
state_dict = torch.load("pytorch_model.bin", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
```

### Loading Tokenizer

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("./")
tokenizer.pad_token = tokenizer.eos_token
```

### Text Generation

```python
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Example
output = generate(model, tokenizer, "The future of AI is", max_new_tokens=50)
print(output)
```

## Files

| File | Description | Size |
|------|-------------|------|
| `model.py` | Model architecture code | ~5KB |
| `config.json` | Model configuration | <1KB |
| `tokenizer.json` | Tokenizer vocabulary | ~3.3MB |
| `tokenizer_config.json` | Tokenizer configuration | <1KB |
| `special_tokens_map.json` | Special tokens mapping | <1KB |

## Model Weights

The pre-trained model weights (`pytorch_model.bin`, ~4GB) are not included in this repository due to size limitations.

**Download options:**
- Contact the author for access to the model weights
- The SFT (fine-tuned) version will be released separately

## Limitations

- This is a **base model** (pre-trained only), not instruction-tuned
- May generate biased or harmful content
- Not optimized for specific downstream tasks without fine-tuning
- Korean language understanding is limited compared to English

## Future Work

- SFT (Supervised Fine-Tuning) for instruction following
- RLHF for improved alignment
- Multilingual expansion

## License

MIT License

## Citation

```bibtex
@misc{ondi2024,
  author = {Junhu Han},
  title = {OnDI v2: Korean-English Bilingual Language Model},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/junhuhan99/OnDI_v2}
}
```

## Acknowledgments

- Trained using DeepSpeed ZeRO-3 optimization
- Built with PyTorch and Hugging Face Transformers
