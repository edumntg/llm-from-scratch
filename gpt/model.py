import torch
import torch.nn as nn
import tiktoken
from simple_parsing.helpers import Serializable
from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelArgs(Serializable):
    emb_dim: int                            # Embedding dimension
    vocab_size: int                         # Vocab size
    context_length: int                     # Context length/window
    num_heads: int                          # Num of attention heads
    num_blocks: int                         # Num of transformer blocks

    dropout: Optional[float] = 0.1          # Dropout rate
    ff_dropout: Optional[float] = 0.1       # Feed-forward's dropout rate
    qkv_bias: Optional[bool] = False        # Bias on QKV parameters
    norm_eps: Optional[float] = 1e-5        # Normalization epsilon (use really small values)

class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        emb_dim, qkv_bias, context_length, dropout = args.emb_dim, args.qkv_bias, args.context_length, args.dropout

        assert emb_dim % args.num_heads == 0, "emb_dim must be divisible by num_heads"

        self.head_dim = emb_dim // args.num_heads
        self.num_heads = args.num_heads



        self.wq = nn.Linear(emb_dim, emb_dim, qkv_bias)
        self.wk = nn.Linear(emb_dim, emb_dim, qkv_bias)
        self.wv = nn.Linear(emb_dim, emb_dim, qkv_bias)

        self.proj = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

        # Mask
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, emb_dim = x.shape

        # Compute QKV values
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        # Current shape of Q, K, V: (batch_size, num_tokens, num_heads, head_dim)

        # Reshape QKV
        Q = Q.view(batch_size, self.num_heads, num_tokens, self.head_dim)
        K = K.view(batch_size, self.num_heads, num_tokens, self.head_dim)
        V = V.view(batch_size, self.num_heads, num_tokens, self.head_dim)

        # Q, K and V have dimesions (batch_size, num_heads, num_tokens, head_dim)

        # Compute attention scores
        attn_scores = Q @ K.transpose(2,3) # (..., num_tokens, head_dim)@(..., head_dim, num_tokens) = (..., num_tokens, num_tokens)

        # Apply mask
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # READ AGAIN IN CHAPTER 2

        # Now compute weights
        d_k = K.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k ** 0.5, dim = -1)
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Finally, compute context vectors
        context_vectors = attn_weights @ V # (..., num_tokens, num_tokens)@(..., num_tokens, head_dim) = (..., num_tokens, head_dim)

        # context_vectors has shape (batch_size, num_heads, num_tokens, head_dim)
        #  Reshape to original shape: (batch_size, num_tokens, num_heads, head_dim)
        context_vectors = context_vectors.transpose(1,2)

        # Now, combine heads into a single tensor of num_heads*head_dim columns
        context_vectors = context_vectors.contiguous().view(batch_size, num_tokens, emb_dim) # NOTE: emb_dim = num_heads*head_dim

        # Apply linear layer (projection)
        context_vectors = self.proj(context_vectors)

        return context_vectors

# Implementation of the FF block
class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2*torch.pi))) * (x + 0.044715 * torch.pow(x, 3)))

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(args.emb_dim, 4 * args.emb_dim),
            GeLU(),
            nn.Linear(4 * args.emb_dim, args.emb_dim),
            nn.Dropout(args.ff_dropout)
        )

    def forward(self, x):
        return self.model(x)

# LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.eps = args.norm_eps
        self.scale = nn.Parameter(torch.ones(args.emb_dim))
        self.bias = nn.Parameter(torch.ones(args.emb_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True)
        return self.scale * ((x - mean) / torch.sqrt(var + self.eps)) + self.bias
    
# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.norm1 = LayerNorm(args)
        self.attention = MultiHeadAttention(args)
        self.dropout1 = nn.Dropout(args.dropout)
        self.norm2 = LayerNorm(args)
        self.feedforward = FeedForward(args)
        self.dropout2 = nn.Dropout(args.dropout)


    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout1(x)

        # Now, add shortcut
        x = x + shortcut

        # Update shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.dropout2(x)

        x = x + shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.token_embedding = nn.Embedding(args.vocab_size, args.emb_dim) # Token embedding
        self.pos_embedding = nn.Embedding(args.context_length, args.emb_dim) # Positional Embedding

        self.transformer = nn.Sequential(
            *[TransformerBlock(args) for _ in range(args.num_blocks)]
        )

        self.final_norm = LayerNorm(args)
        self.output = nn.Linear(args.emb_dim, args.vocab_size)

    def forward(self, inputs):
        # X is a tokenized vector
        batch_size, seq_len = inputs.shape
        tok_emb = self.token_embedding(inputs)

        pos_emb = self.pos_embedding(torch.arange(seq_len, device = inputs.device))

        x = tok_emb + pos_emb

        x = self.transformer(x)

        x = self.final_norm(x)

        logits = self.output(x)

        return logits