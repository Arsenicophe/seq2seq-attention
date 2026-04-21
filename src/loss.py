"""
Loss seq2seq avec gestion du padding.

CrossEntropyLoss avec ignore_index=PAD_IDX :
les positions paddées ne contribuent pas au gradient.

Convention d'appel :
    logits = model(src, src_lengths, trg[:-1], trg_lengths - 1)
    loss   = criterion(logits, trg[1:])
"""

import torch
import torch.nn as nn
from data import PAD_IDX


class Seq2SeqLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0):
        """
        Args:
            label_smoothing : régularisation — évite que le modèle soit trop confiant (0 = désactivé)
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=PAD_IDX,
            label_smoothing=label_smoothing,
            reduction="mean",
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (trg_len, batch, vocab_size)
            target : (trg_len, batch)

        Returns:
            loss : scalaire — moyenne sur les tokens non paddés
        """
        return self.criterion(
            logits.flatten(0, 1),   # (trg_len * batch, vocab_size)
            target.flatten(0, 1),   # (trg_len * batch,)
        )
