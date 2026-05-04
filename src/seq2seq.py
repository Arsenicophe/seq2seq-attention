"""
Assemblage Encoder + Decoder.
Le lien entre les deux passe uniquement par enc_outputs (cross-attention).
"""

import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

PAD_IDX = 0


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    def forward(
        self,
        src         : torch.Tensor,   # (src_len, batch)
        src_lengths : torch.Tensor,   # (batch,)
        trg         : torch.Tensor,   # (trg_len, batch)  — trg[:-1] à l'entraînement
        trg_lengths : torch.Tensor,   # (batch,)
    ) -> torch.Tensor:
        """
        Returns:
            logits : (trg_len, batch, vocab_size)
        """
        enc_outputs, h_n, c_n = self.encoder(src, src_lengths)

        # Masque (batch, src_len) — True aux positions PAD, ignorées par le MHA
        src_key_padding_mask = (src == PAD_IDX).T.to(self.device)

        # h_n, c_n initialisent LSTM 1 du décodeur (transfert encodeur → décodeur)
        return self.decoder(trg, trg_lengths, enc_outputs, h_n, c_n,
                            src_key_padding_mask=src_key_padding_mask)
