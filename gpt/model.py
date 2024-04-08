import torch
import torch.nn as nn
import tiktoken
from simple_parsing.helpers import Serializable
from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelArgs(Serializable):
    emb_dim: int
    context_length: int
    num_heads: int

    dropout: Optional[float] = 0.1
    ff_dropout: Optional[float] = 0.1
    qkv_bias: Optional[bool] = False

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
        print(attn_scores.shape)
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

