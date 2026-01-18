"""
OnDI v2 - 983M Parameter Language Model Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OndiMLP(nn.Module):
    """Feed-Forward Network with SwiGLU activation"""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class OndiAttention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, hidden_size, num_heads, max_position=2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, attention_mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask
        causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class OndiBlock(nn.Module):
    """Transformer Decoder Block"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config["hidden_size"])
        self.attn = OndiAttention(
            config["hidden_size"],
            config["num_heads"],
            config["max_position_embeddings"]
        )
        self.ln2 = nn.LayerNorm(config["hidden_size"])
        self.mlp = OndiMLP(config["hidden_size"], config["intermediate_size"])

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class OndiModel(nn.Module):
    """
    OnDI v2 Language Model

    983M parameter decoder-only transformer with:
    - Learned positional embeddings
    - Pre-LN architecture
    - SwiGLU activation
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.embed_positions = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.layers = nn.ModuleList([OndiBlock(config) for _ in range(config["num_layers"])])
        self.norm = nn.LayerNorm(config["hidden_size"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        self.gradient_checkpointing = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config["vocab_size"]),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, top_p=0.9):
        """Generate text autoregressively"""
        self.eval()

        for _ in range(max_new_tokens):
            logits = self(input_ids)["logits"][:, -1, :]
            logits = logits / temperature

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids


# Default configuration
DEFAULT_CONFIG = {
    "vocab_size": 50257,
    "max_position_embeddings": 2048,
    "hidden_size": 1536,
    "num_layers": 24,
    "num_heads": 16,
    "intermediate_size": 6144,
}


def create_model(config=None):
    """Create OnDI model with default or custom config"""
    if config is None:
        config = DEFAULT_CONFIG
    return OndiModel(config)


def load_model(checkpoint_path, config=None, device="cpu"):
    """Load OnDI model from checkpoint"""
    model = create_model(config)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    params = sum(p.numel() for p in model.parameters())
    print(f"OnDI v2 Model")
    print(f"Parameters: {params:,} ({params/1e6:.1f}M)")

    # Test forward pass
    dummy_input = torch.randint(0, 50257, (1, 128))
    output = model(dummy_input)
    print(f"Output shape: {output['logits'].shape}")
