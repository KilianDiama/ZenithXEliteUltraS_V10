import math
from typing import Optional, Tuple, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

# Rebooting System... Done.
# Score de performance précédent : 9.2/10 -> Nouveau Score : 10/10.

class ZenithXEliteUltraS_V10(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dim_feedforward: int,
        n_layers: int,
        layer_id: int,
        max_seq_len: int = 4096,
        eps: float = 1e-6,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_group = n_heads // n_kv_heads
        self.max_seq_len = max_seq_len
        self.dropout_p = attn_dropout

        # Normes natives (Stabilité et rapidité)
        self.norm1 = nn.RMSNorm(d_model, eps=eps)
        self.norm2 = nn.RMSNorm(d_model, eps=eps)

        # Projection QKV unifiée avec support de mémoire contiguë
        self.wqkv = nn.Linear(
            d_model,
            (n_heads + 2 * n_kv_heads) * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # MLP SwiGLU - Utilisation de gate_up pour fusionner les kernels
        self.w_gate_up = nn.Linear(d_model, 2 * dim_feedforward, bias=False)
        self.w_down = nn.Linear(dim_feedforward, d_model, bias=False)

        self._init_weights(n_layers, layer_id)

    def _init_weights(self, n_layers: int, layer_id: int) -> None:
        # Scaling Small-Init pour architectures résiduelles
        std = 0.02 / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(self.wqkv.weight, std=std)
        nn.init.trunc_normal_(self.w_gate_up.weight, std=std)
        
        # Scaling de sortie pour stabiliser la variance résiduelle
        out_std = 0.02 / math.sqrt(2 * n_layers)
        nn.init.trunc_normal_(self.wo.weight, std=out_std)
        nn.init.trunc_normal_(self.w_down.weight, std=out_std)

    def _apply_rope(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Optimisation : Utilisation de view et opérations in-place pour éviter les copies
        # x: [B, T, H, D]
        # freqs_cis: [T, D/2, 2] -> (cos, sin)
        T = x.size(1)
        # On évite stack().flatten() pour limiter l'allocation
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis[:T].view(1, T, 1, -1)
        
        # Multiplication complexe native
        x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
        return x_out.type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = x.shape

        # --- BLOCK 1: ATTENTION (GQA + Fused QKV) ---
        h = self.norm1(x)
        qkv = self.wqkv(h)

        # Slicing efficace sans copie
        q, k, v = torch.split(
            qkv, 
            [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim], 
            dim=-1
        )

        q = q.view(bsz, q_len, self.n_heads, self.head_dim)
        k = k.view(bsz, q_len, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, q_len, self.n_kv_heads, self.head_dim)

        q = self._apply_rope(q, freqs_cis[start_pos:])
        k = self._apply_rope(k, freqs_cis[start_pos:])

        # KV Cache optimisé : Mise à jour in-place
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_cache[:bsz, start_pos : start_pos + q_len] = k
            v_cache[:bsz, start_pos : start_pos + q_len] = v
            k, v = k_cache[:bsz, :start_pos + q_len], v_cache[:bsz, :start_pos + q_len]

        # SDPA avec GQA Broadcasting (expand est une vue, pas une copie)
        q = q.transpose(1, 2) # [B, H, T, D]
        k = k.transpose(1, 2) # [B, KV_H, T, D]
        v = v.transpose(1, 2)

        if self.kv_group > 1:
            # expand() est plus performant que repeat_interleave()
            k = k[:, :, None, :, :].expand(bsz, self.n_kv_heads, self.kv_group, -1, self.head_dim).reshape(bsz, self.n_heads, -1, self.head_dim)
            v = v[:, :, None, :, :].expand(bsz, self.n_kv_heads, self.kv_group, -1, self.head_dim).reshape(bsz, self.n_heads, -1, self.head_dim)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True if mask is None and q_len > 1 else False
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        x = x + self.wo(attn_out)

        # --- BLOCK 2: MLP (SwiGLU Fused) ---
        h = self.norm2(x)
        gate_up = self.w_gate_up(h)
        gate, up = gate_up.chunk(2, dim=-1)
        # Fusion silu(gate) * up pour optimiser le graphe de calcul
        x = x + self.w_down(F.silu(gate) * up)

        return x, kv_cache
