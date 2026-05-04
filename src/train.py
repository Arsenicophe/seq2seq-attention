"""
Boucles d'entraînement et d'évaluation.

Points clés :
- trg[:-1] → entrée décodeur  (SOS, w1, ..., wn)
- trg[1:]  → cible loss        (w1, ..., wn, EOS)
- src_lengths et trg_lengths restent sur CPU (requis par pack_padded_sequence)
- gradient clipping à 1.0 — stabilise l'entraînement des RNN
- scaler (GradScaler) : activer pour mixed precision (FP16) sur GPU → ~2× moins de mémoire
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from seq2seq import Seq2Seq


def train_epoch(
    model    : Seq2Seq,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    clip     : float = 1.0,
    device   : torch.device = torch.device("cpu"),
    scaler   : GradScaler | None = None,
) -> float:
    """
    Returns:
        loss moyenne sur l'époque
    """
    model.train()
    total_loss = 0.0

    for src, src_lengths, trg, trg_lengths in loader:
        src, trg = src.to(device), trg.to(device)
        # src_lengths et trg_lengths restent sur CPU

        optimizer.zero_grad()

        # autocast active FP16 sur GPU ; no-op sur CPU
        with autocast(device_type=device.type, enabled=scaler is not None):
            logits, _ = model(src, src_lengths, trg[:-1], trg_lengths - 1)
            loss       = criterion(logits, trg[1:])

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model    : Seq2Seq,
    loader,
    criterion: nn.Module,
    device   : torch.device = torch.device("cpu"),
) -> float:
    """
    Returns:
        loss moyenne sur le jeu d'évaluation
    """
    model.eval()
    total_loss = 0.0

    for src, src_lengths, trg, trg_lengths in loader:
        src, trg = src.to(device), trg.to(device)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits, _ = model(src, src_lengths, trg[:-1], trg_lengths - 1)
            loss       = criterion(logits, trg[1:])

        total_loss += loss.item()

    return total_loss / len(loader)
