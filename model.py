import os
import math
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
TODO:
- training loop
- generation
"""


def dropout(x: torch.Tensor, rate: float, is_training=True):
    if is_training and rate > 0:
        return F.dropout(x, p=rate, training=is_training)
    else:
        return x


class Linear(nn.Module):
    def __init__(self, 
                in_dim: int, 
                out_dim: int, 
                bias: bool = True,
                name: str = None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)
        if bias:
            self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ret = torch.einsum('io, ...i -> ...o', self.weight, x)
        if self.bias:
            ret += self.b
        return ret


class RMSNorm(nn.Module):
    def __init__(self, 
                dim: int,
                eps: float = 1e-5, 
                name: str = None) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        
        mean_squared = torch.mean(torch.square(x), dim=-1, keepdim=True)
        x = x / torch.sqrt(mean_squared + self.eps)
        x = self.scale * x
    
        return x.to(input_dtype)


class DenseFF(nn.Module):
    def __init__(self,
                emd_dim: int, 
                hidden_dim: int, 
                activation: str = 'gelu', 
                bias: bool = True,
                name: str = None) -> None:
        super().__init__()
        self.w1 = Linear(emd_dim, hidden_dim, bias=bias)
        self.w2 = Linear(emd_dim, hidden_dim, bias=bias)
        self.w3 = Linear(hidden_dim, emd_dim, bias=bias)

        if activation not in ('gelu', 'silu', 'relu'):
            raise ValueError(f'Unknown activation function: {activation}')
            
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'silu':
            self.activation = F.silu
        elif activation == 'relu':
            self.activation = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w2(x) * self.w1(x)
        h = self.activation(h)
        return self.w3(h)


class RotaryEmbedding(nn.Module):
    def __init__(self, 
                dim,
                max_seq_len: int = 8000,
                base: int = 10000,
                name: str = None) -> None:
        super().__init__()
        assert dim % 2 == 0
        
        # thetas: D/2
        exps = -torch.arange(0, dim, 2, dtype=torch.float32) / dim
        thetas = base ** exps
        
        t = torch.arange(0, max_seq_len, dtype=torch.float32)
        # ticks: t x D/2
        ticks = torch.outer(t, thetas)
        # ticks: t x D/2 -> 1 x t x 1 x D 
        ticks = torch.tile(ticks, (1, 2)).unsqueeze(0).unsqueeze(2)
        
        # cos, sin: 1 x t x 1 x D
        self.register_buffer('cos', torch.cos(ticks))
        self.register_buffer('sin', torch.sin(ticks))

    def _neg_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, x.shape[-1] // 2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
          x: B x t x H x D
        """
        _, seq_len, _, D = x.shape
        
        cos = self.cos[:, offset:offset+seq_len]
        sin = self.sin[:, offset:offset+seq_len]
        
        rote = x * cos + self._neg_half(x) * sin
        
        return rote


class MultiHeadAttention(nn.Module):
    def __init__(self, 
        num_q_heads, 
        num_kv_heads,
        emd_dim: int,
        v_dim: int,
        k_dim: int,
        bias: bool = False,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        name: str = None,
    ) -> None:
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        assert attn_dropout >= 0 and resid_dropout >= 0

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.q_group_size = num_q_heads // num_kv_heads
        self.emd_dim = emd_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.bias = bias

        self.wq = Linear(self.emd_dim, self.k_dim * self.num_q_heads, bias=self.bias)
        self.wk = Linear(self.emd_dim, self.k_dim * self.num_kv_heads, bias=self.bias)
        self.wv = Linear(self.emd_dim, self.v_dim * self.num_kv_heads, bias=self.bias)
        self.wo = Linear(self.q_group_size * self.num_kv_heads * self.v_dim, self.emd_dim, 
          bias=self.bias
        )

        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout
        self.rote = RotaryEmbedding(self.k_dim)

    # NOTE: Even for self attention, accepting three separate arguments is faster with jit
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_cache: Optional[torch.Tensor] = None, is_training=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            B: batch size
            t: length of the input sequence
            T: length of the attended sequence
            D: embedding dimension
            K=Q: key/query dimension
            V: value dimension
            h: number of query heads
            H: number of key/value heads
            G: number of query groups

            q, k, v: B x t x D
            kv_cache: B x T x H x (K+V)
        """

        # q_heads, k_heads: B x t x h x K
        # v_heads: B x T x H x V
        q_heads, k_heads, v_heads = self._attention_heads(q, k, v, kv_cache)
        attn_output = self._grouped_attention(q_heads, k_heads, v_heads, is_training) # B x t x D
        new_kv_cache = torch.cat([k_heads, v_heads], dim=-1)
        
        return attn_output, new_kv_cache

    def _grouped_attention(self, q_heads: torch.Tensor, k_heads: torch.Tensor, v_heads: torch.Tensor, is_training) -> torch.Tensor:
        b_size, q_seq_len, _, _ = q_heads.shape
        _, k_seq_len, _, _ = k_heads.shape

        grouped_q_heads = q_heads.reshape(b_size, q_seq_len, self.q_group_size, self.num_kv_heads, self.k_dim)

        attn_scores = torch.einsum('BtGHK, BTHK-> BGHtT', grouped_q_heads, k_heads).to(torch.float32)
        attn_mask = torch.tril(torch.ones(k_seq_len, k_seq_len, device=q_heads.device))[-q_seq_len:, :]
        attn_scores = torch.where(attn_mask.bool(), attn_scores, torch.tensor(-float('inf'), device=q_heads.device))

        attn_scores = F.softmax(attn_scores / (self.k_dim ** 0.5), dim=-1).to(v_heads.dtype)
        attn_scores = dropout(attn_scores, self.attn_dropout, is_training)

        attn_output = torch.einsum("BGHtT, BTHV -> BtGHV", attn_scores, v_heads)
        attn_output = attn_output.reshape(b_size, q_seq_len, self.q_group_size * self.num_kv_heads * self.v_dim)

        attn_output = self.wo(attn_output)
        attn_output = dropout(attn_output, self.resid_dropout, is_training)

        return attn_output

    def _attention_heads(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_cache: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
            q, k, v: B x t x D
            kv_cache: B x T x H x (K+V)
        """
        b_size, seq_len, _ = q.shape
        offset = 0 if kv_cache is None else kv_cache.shape[1]

        # q_heads: B x t x h x Q=K
        q_heads = self.wq(q).reshape(b_size, seq_len, self.num_q_heads, self.k_dim)
        q_heads = self.rote(q_heads, offset)
        # k_heads: B x T x H x K
        k_heads = self.wk(k).reshape(b_size, seq_len, self.num_kv_heads, self.k_dim)
        k_heads = self.rote(k_heads, offset)
        # v_heads: B x T x H x V
        v_heads = self.wv(v).reshape(b_size, seq_len, self.num_kv_heads, self.v_dim)
        
        # stack newly computed heads with cached key and value heads to create the the full key and value heads
        if kv_cache is not None:
            b_size_cached, seq_len_cached, num_kv_heads_cached, kv_dim_cached = kv_cache.shape
            assert (b_size_cached, num_kv_heads_cached, kv_dim_cached) == (b_size, self.num_kv_heads, self.k_dim + self.v_dim)
            
            k_heads_cached = kv_cache[:, :, :, :self.k_dim]
            v_heads_cached = kv_cache[:, :, :, self.k_dim:]

            k_heads = torch.cat([k_heads_cached, k_heads], dim=1)
            v_heads = torch.cat([v_heads_cached, v_heads], dim=1)

        return q_heads, k_heads, v_heads


class MoEBlock(nn.Module):
    def __init__(self,
        emd_dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        active_experts: int = 2,
        expert_capacity: float = 1.0,
        ff_bias: bool = False,
        name: str = None
    ):
        super().__init__()
        self.expert_capacity = expert_capacity
        self.top_k = active_experts
        
        self.emd_dim = emd_dim
        self.hidden_dim = hidden_dim
        self.ff_bias = ff_bias
        self.num_experts = num_experts
        self.router = Linear(emd_dim, num_experts)
        
        # Create expert parameters
        self.experts = nn.ModuleList([
            DenseFF(emd_dim, hidden_dim, bias=ff_bias) 
            for _ in range(num_experts)
        ])
    
    def _compute_expert_scores(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        expert_scores = self.router(x_flat.to(torch.float32)) # (B * t) x num_experts
        expert_scores, expert_assignment = torch.topk(expert_scores, k=self.top_k, dim=-1) # (B * t) x top_k
        expert_scores = F.softmax(expert_scores, dim=-1).to(x_flat.dtype) # (B * t) x top_k

        return expert_assignment, expert_scores

    def _compute_token_expert_assignment(self, x_flat: torch.Tensor, expert_assignment: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            B: batch size
            t: length of the input sequence
            D: embedding dimension

            x_flat: (B * t) x D
        """
        
        B_t, _ = x_flat.shape
        device = x_flat.device
        
        # expert capacity is the fraction of the total number of tokens that can be routed to an expert
        expert_capacity = math.floor(B_t * self.expert_capacity)
        # For each token, each choice, and each expert, there is either an assignment of the token to the expert or not
        expert_assignment_onehot = F.one_hot(expert_assignment, self.num_experts).float() # (B * t) x top_k x num_experts
        
        # Cumulative sum over the token indices, and then cumulative sum over the top_k dimension
        position_in_experts = torch.cumsum(torch.cumsum(expert_assignment_onehot, dim=0), dim=1).to(torch.int32) # (B * t) x top_k x num_experts
        
        # Using the assigned positions, remove any token that did not make it within an expert's capacity
        expert_mask = expert_assignment_onehot * (position_in_experts < expert_capacity) # (B * t) x top_k x num_experts
        # By summing over the experts dimension, obtain a mask of which tokens are processed by some experts and which ones are orphans
        token_assignment_mask = torch.sum(expert_mask, dim=-1) # (B * t) x top_k
        
        # Apply onehot operate to the position array
        expert_choices = F.one_hot(position_in_experts, expert_capacity).float() * expert_mask.unsqueeze(-1) # (B * t) x top_k x num_experts x expert_capacity
    
        # Sum over the top_k dimension
        expert_choices = torch.sum(expert_choices, dim=1) # (B * t) x num_experts x expert_capacity
        expert_choices = expert_choices.permute(1, 2, 0).to(torch.int32) # num_experts x expert_capacity x (B * t)
        
        # For each token, and for each choice index, which expert it gets assigned to
        expert_assignment = torch.einsum('tke, e -> tk', expert_mask, torch.arange(self.num_experts, device=device)) # (B * t) x top_k
        # Extract out the position in the selected expert
        position_in_selected_experts = torch.einsum('tke, tke -> tk', expert_mask, position_in_experts.float()) # (B * t) x top_k
        # Stack them to a single tensor
        expert_position_assignment = torch.stack([expert_assignment, position_in_selected_experts], dim=-1).to(torch.int32) # (B * t) x top_k x 2
        
        return expert_choices, expert_position_assignment, token_assignment_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """      
            B: batch size
            t: length of the input sequence
            D: embedding dimension

            x: B x t x D
        """
        b, t, D = x.shape
        device = x.device
        x_flat = x.reshape(b * t, D) 
        
        # expert_assignment, expert_scores: (B * t) x top_k
        expert_assignment, expert_scores = self._compute_expert_scores(x_flat)
        
        # expert_choices: num_experts x expert_capacity x (B * t)
        # expert_position_assignment: (B * t) x top_k x 2
        # token_assignment_mask: (B * t) x top_k
        expert_choices, expert_position_assignment, token_assignment_mask = self._compute_token_expert_assignment(x_flat, expert_assignment)
        
        # Reshape to have a batch dimension
        expert_scores = torch.reshape(expert_scores, (b, t, self.top_k)) # B x t x top_k
        expert_position_assignment = torch.reshape(expert_position_assignment, (b, t, self.top_k, 2)) # B x t x top_k x 2
        token_assignment_mask = torch.reshape(token_assignment_mask, (b, t, self.top_k)) # B x t x top_k
        
        # Initialize expert outputs tensor
        expert_outputs = torch.zeros(b, t, self.top_k, D, device=device).to(x.dtype)
        
        # Process tokens with experts
        for i, expert in enumerate(self.experts):
            # Get tokens assigned to this expert
            expert_mask = (expert_position_assignment[..., 0] == i)
            if not torch.any(expert_mask):
                continue
                
            # Get positions within this expert
            positions = expert_position_assignment[expert_mask][..., 1].long()
            
            # Get batch and sequence indices for these tokens
            batch_indices = torch.arange(b, device=device).view(b, 1, 1).expand(b, t, self.top_k)[expert_mask]
            seq_indices = torch.arange(t, device=device).view(1, t, 1).expand(b, t, self.top_k)[expert_mask]
            
            # Process tokens with this expert
            token_inputs = x[batch_indices, seq_indices]
            token_outputs = expert(token_inputs)
            
            # Assign outputs back to the correct positions
            expert_outputs[batch_indices, seq_indices] = token_outputs.unsqueeze(1)
            
        # Dropped tokens will receive a residual connection
        expert_outputs = torch.where(token_assignment_mask.unsqueeze(-1).bool(), expert_outputs, x.unsqueeze(2))
        result = torch.einsum('btkD, btk -> btD', expert_outputs, expert_scores)
        
        return result


class TransformerBlock(nn.Module):
    def __init__(self, 
        emd_dim: int,
        num_q_heads, 
        num_kv_heads,
        v_dim: int,
        k_dim: int,
        hidden_dim: int,
        num_experts: int,
        active_experts: int,
        expert_capacity: int,
        ff_bias: bool = False,
        multi_device: bool = True,
        attn_bias: bool = False,
        attn_dropout: float = 0.1,
        attn_resid_dropout: float = 0.0,
        name: str = None,
    ) -> None:
        super().__init__()
        self.pre_layer_norm = RMSNorm(emd_dim)
        self.post_attn_norm = RMSNorm(emd_dim)
        self.emd_dim = emd_dim

        self.attn = MultiHeadAttention(
            num_q_heads, 
            num_kv_heads,
            emd_dim,
            v_dim,
            k_dim,
            bias=attn_bias,
            attn_dropout=attn_dropout,
            resid_dropout=attn_resid_dropout
        )

        self.num_experts = num_experts
        if num_experts > 1:
            self.moe = MoEBlock(
                emd_dim, 
                hidden_dim,
                num_experts,
                active_experts,
                expert_capacity,
                ff_bias
            )
        else: 
            self.ff = DenseFF(emd_dim, hidden_dim, bias=ff_bias)

    def forward(self, x: torch.Tensor, kv_cache: Optional[torch.Tensor] = None, is_training=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """      
            B: batch size
            t: length of the input sequence
            D: embedding dimension
            K=Q: key/query dimension
            V: value dimension
            H: number of key/value heads

            x: B x t x D
            kv_cache: B x T x H x (K+V)
        """

        # Normalize before attention
        h = self.pre_layer_norm(x)
        h, new_kv_cache = self.attn(h, h, h, kv_cache, is_training)
        # Residual connection
        h = h + x 
        # Post residual normalization
        r = self.post_attn_norm(h)
        if self.num_experts > 1:
            r = self.moe(r)
        else:
            r = self.ff(r)

        return r, new_kv_cache
    

class Embedding(nn.Module):
    def __init__(self, 
                emd_dim: int, 
                n_vocab: int,
                name: str = None) -> None:
        super().__init__()
        self.emd_dim = emd_dim
        self.vocab_size = n_vocab
        self.embedding = nn.Parameter(torch.randn(n_vocab, emd_dim) * 0.02)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('ve, ...v -> ...e', self.embedding, x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('ve, ...e -> ...v', self.embedding, x)


class MoeTransformer(nn.Module):
    def __init__(self,
        depth: int, 
        n_vocab: int,
        emd_dim: int,
        num_q_heads, 
        num_kv_heads,
        v_dim: int,
        k_dim: int,
        hidden_dim: int,
        num_experts: int,
        active_experts: int,
        expert_capacity: int,
        ff_bias: bool = False,
        multi_device: bool = True,
        attn_bias: bool = False,
        attn_dropout: float = 0.0,
        attn_resid_dropout: float = 0.0,
        name: str = None,
    ):
        super().__init__()

        self.block_config = {
            'emd_dim': emd_dim,
            'num_q_heads': num_q_heads,
            'num_kv_heads': num_kv_heads,
            'v_dim': v_dim,
            'k_dim': k_dim,
            'hidden_dim': hidden_dim,
            'num_experts': num_experts,
            'active_experts': active_experts,
            'expert_capacity': expert_capacity,
            'ff_bias': ff_bias,
            'attn_bias': attn_bias,
            'attn_dropout': attn_dropout,
            'attn_resid_dropout': attn_resid_dropout,
        }
        
        self.embedding = Embedding(emd_dim, n_vocab)
        self.final_norm = RMSNorm(emd_dim)
        
        # Create transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(**self.block_config) for _ in range(depth)
        ])

        self.num_experts = num_experts
        self.depth = depth
        self.emd_dim = emd_dim
        self.n_vocab = n_vocab
    
    def forward(self, x: torch.Tensor, kv_caches: Dict[int, torch.Tensor] = None, is_training=True) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
            B: batch size
            t: length of the input sequence
            T: length of the attended sequence
            D: embedding dimension
            K=Q: key/query dimension
            V: value dimension
            H: number of key/value heads

            x: B x t (token ids)
            kv_caches[i]: B x T x H x (K+V)
        """
        if kv_caches is None:
            kv_caches = {}
            
        # Convert token ids to one-hot encoding if not already done
        if x.dim() == 2:
            x = F.one_hot(x, self.n_vocab).float()
            
        # Embed input
        h = self.embedding.encode(x)

        new_kv_caches = {}
        for i, block in enumerate(self.blocks):
            h, new_kv_cache = block(h, kv_caches.get(i, None), is_training)
            new_kv_caches[i] = new_kv_cache

        h = self.final_norm(h)
        r = self.embedding.decode(h)

        return r, new_kv_caches