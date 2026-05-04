"""
Encodeur BiLSTM.
Lit la séquence source et produit enc_outputs (K, V pour la cross-attention du décodeur).
Le BiLSTM lit la séquence dans les deux sens — chaque position encode son contexte gauche ET droit.
La projection ramène hidden*2 → hidden pour être compatible avec le décodeur.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.rnn        = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim,
                                  num_layers=n_layers, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src        : (src_len, batch)
            src_lengths: (batch,)  longueurs réelles — les PAD sont ignorés par le LSTM

        Returns:
            enc_outputs : (src_len, batch, hidden_dim)
        """
        x = self.dropout(self.embedding(src))

        packed = pack_padded_sequence(x, src_lengths.cpu(), enforce_sorted=False)
        packed_out, (h_n, c_n) = self.rnn(packed)
        enc_outputs, _ = pad_packed_sequence(packed_out, total_length=src.size(0))

        # BiLSTM : h_n shape (n_layers*2, batch, hidden)
        # On fusionne forward + backward pour chaque couche → (n_layers, batch, hidden)
        h_n = self.projection(torch.cat([h_n[0::2], h_n[1::2]], dim=-1))
        c_n = self.projection(torch.cat([c_n[0::2], c_n[1::2]], dim=-1))

        return self.projection(enc_outputs), h_n, c_n
