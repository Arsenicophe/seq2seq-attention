"""
Décodeur : LSTM 1 → Cross-Attention → LSTM 2.

Le décodeur traite la séquence cible entière d'un coup (teacher forcing total à l'entraînement).

  trg (trg_len, batch)
       │
  [Embedding + Dropout]
       │
  [LSTM 1]             — lit les tokens cibles décalés (<SOS>, w1, w2, ...)
       │  outputs1
  [CrossAttention]     — Q=outputs1, K=V=enc_outputs, ignore les PAD source
       │  contexts
  [LSTM 2]             — lit les vecteurs de contexte
       │  outputs2
  [Linear]
       │
  logits (trg_len, batch, vocab_size)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from attention import CrossAttention


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 n_layers: int, n_heads: int, dropout: float, return_state: bool = False):
        """
        Args:
            n_heads      : têtes MHA — doit diviser hidden_dim
            return_state : si True, retourne aussi (h1, c1) du LSTM 1 (utile à l'inférence)
        """
        super().__init__()
        self.embedding    = nn.Embedding(vocab_size, embed_dim)
        self.lstm1        = nn.LSTM(embed_dim,  hidden_dim, n_layers)
        self.attention    = CrossAttention(d_model=hidden_dim, n_heads=n_heads, dropout=dropout)
        self.lstm2        = nn.LSTM(hidden_dim, hidden_dim, n_layers)
        self.fc_out       = nn.Linear(hidden_dim, vocab_size)
        self.dropout      = nn.Dropout(dropout)
        self.return_state = return_state

    def forward(
        self,
        trg                 : torch.Tensor,            # (trg_len, batch)
        trg_lengths         : torch.Tensor,            # (batch,)
        enc_outputs         : torch.Tensor,            # (src_len, batch, hidden_dim)
        h1                  = None,                    # (n_layers, batch, hidden_dim)  état initial LSTM 1
        c1                  = None,                    # (n_layers, batch, hidden_dim)
        src_key_padding_mask: torch.Tensor | None = None,  # (batch, src_len)
    ):
        """
        Returns:
            logits     : (trg_len, batch, vocab_size)
            (h1, c1)   : état final LSTM 1  — seulement si return_state=True
        """
        trg_len = trg.size(0)

        x = self.dropout(self.embedding(trg))                   # (trg_len, batch, embed_dim)

        # LSTM 1 — lit les tokens cibles, produit des représentations contextualisées
        packed1 = pack_padded_sequence(x, trg_lengths.cpu(), enforce_sorted=False)
        packed_out1, (h1, c1) = self.lstm1(packed1, (h1, c1))
        outputs1, _ = pad_packed_sequence(packed_out1, total_length=trg_len)

        # Cross-Attention — chaque position cible consulte toute la source
        contexts = self.attention(outputs1, enc_outputs,
                                  key_padding_mask=src_key_padding_mask)

        # LSTM 2 — intègre le contexte source dans la représentation cible
        packed2 = pack_padded_sequence(contexts, trg_lengths.cpu(), enforce_sorted=False)
        packed_out2, _ = self.lstm2(packed2)
        outputs2, _ = pad_packed_sequence(packed_out2, total_length=trg_len)

        logits = self.fc_out(outputs2)                          # (trg_len, batch, vocab_size)

        if self.return_state:
            return logits, (h1, c1)
        return logits
