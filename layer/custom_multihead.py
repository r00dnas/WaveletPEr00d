import torch 
import torch.nn as nn
import math


class BiasMultiHeadAttention(nn.Module):
    def __init__(self, dim_h, num_heads, attn_dropout, bias = True):
        super().__init__()

        self.dim_h = dim_h 
        self.num_heads = num_heads 
        #self.dropout = dropout
        self.head_dim = dim_h // num_heads
        assert (
            self.head_dim * num_heads == dim_h
        ), "dim_h must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        #self.attn_bias_type = attn_bias_type

        self.q_proj = nn.Linear(dim_h, dim_h, bias=bias)
        self.k_proj = nn.Linear(dim_h, dim_h, bias=bias)
        self.v_proj = nn.Linear(dim_h, dim_h, bias=bias)
        self.out_proj = nn.Linear(dim_h, dim_h, bias=bias)

        self.dropout = nn.Dropout(p=attn_dropout)

        self.edge_mlp = nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU(), nn.Linear(dim_h, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters of projection matrices, the same settings as that in Graphormer."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2**-0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2**-0.5)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, X, E, attn_mask = None):
        q_h = self.q_proj(X).transpose(0, 1)
        k_h = self.k_proj(X).transpose(0, 1)
        v_h = self.v_proj(X).transpose(0, 1)
        bsz, N, _ = X.shape
        q_h = (
            q_h.reshape(bsz, N, self.num_heads, self.head_dim).transpose(1, 2)
        )
        k_h = k_h.reshape(bsz, N, self.num_heads, self.head_dim).transpose(1, 2)
        v_h = v_h.reshape(bsz, N,  self.num_heads, self.head_dim).transpose(1, 2)
        attn_bias = self.edge_mlp(E).permute(0, 3, 1, 2) 
        attn = self.scaled_dot_product_attention(q_h, k_h, v_h, scale = self.scale_factor, attn_bias= attn_bias) 
        attn = attn.transpose(1, 2).reshape(bsz, N, self.head_dim * self.num_heads)
        return self.out_proj(attn), None
    
    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, attn_bias=None):
    # Efficient implementation equivalent to the following:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        if attn_bias is None:
            attn_bias = torch.zeros(L, S, dtype=query.dtype)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor 
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value
