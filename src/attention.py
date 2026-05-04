"""
Cross-Attention via nn.MultiheadAttention.

Q = sorties du LSTM 1 du décodeur  (trg_len, batch, hidden)
K = V = sorties de l'encodeur       (src_len, batch, hidden)

key_padding_mask : (batch, src_len) — True aux positions PAD de la source,
                   ces positions recevront un poids d'attention nul.
"""

import torch
import torch.nn as nn



class CrossAttention(nn.Module):
    """
        Args:
            d_model : dimension Q, K, V  — doit être divisible par n_heads
            n_heads : nombre de têtes
            dropout : dropout interne au MHA
        """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha       = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=False)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)          # dropout sur le résidu (cf. papier)

    def forward(
        self,
        query           : torch.Tensor,
        enc_outputs     : torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        """
        Returns:
            context : (trg_len, batch, hidden)
        """

        context, alpha = self.mha(
            query, enc_outputs, enc_outputs,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        # Add & Norm
        out = self.layernorm(query + self.dropout(context))
        return out, alpha