"""
Pipeline de données : Vocab, Dataset, collate_fn.

Conventions d'index :
    PAD = 0  SOS = 1  EOS = 2  UNK = 3
    Vocabulaire réel à partir de l'index 4.

Flux :
    texte brut → tokenisation → Vocab.encode → TranslationDataset
              → DataLoader(collate_fn) → batch paddé (time-first)
"""

from __future__ import annotations
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter


PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]


class Vocab:
    """Mapping bidirectionnel token ↔ index."""

    def __init__(self, tokens: list[str]):
        self.itos = SPECIAL_TOKENS + tokens          # index → token
        self.stoi = {t: i for i, t in enumerate(self.itos)}  # token → index

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, sentence: list[str]) -> list[int]:
        """Tokens → indices. Tokens inconnus → UNK_IDX."""
        return [self.stoi.get(t, UNK_IDX) for t in sentence]

    def decode(self, indices: list[int]) -> list[str]:
        """Indices → tokens."""
        return [self.itos[i] for i in indices]

    @classmethod
    def build(cls, corpus: list[list[str]], min_freq: int = 1) -> "Vocab":
        """
        Construit le vocabulaire depuis un corpus tokenisé.
        Filtre les tokens apparaissant moins de min_freq fois.
        Trie par fréquence décroissante puis alphabétiquement.
        """
        freqs = Counter(t for sentence in corpus for t in sentence)
        tokens = sorted(
            (t for t, f in freqs.items() if f >= min_freq),
            key=lambda t: (-freqs[t], t),
        )
        return cls(tokens)


class TranslationDataset(Dataset):
    """
    Paires (src_ids, trg_ids) numérotées, sans padding.
    Le padding est appliqué au niveau du batch dans collate_fn.

    Convention :
        src : [w1, w2, ..., <EOS>]
        trg : [<SOS>, w1, w2, ..., <EOS>]
    """

    def __init__(self, src_sentences: list[list[str]], trg_sentences: list[list[str]],
                 src_vocab: Vocab, trg_vocab: Vocab):
        assert len(src_sentences) == len(trg_sentences)
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab     = src_vocab
        self.trg_vocab     = trg_vocab

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src = self.src_vocab.encode(self.src_sentences[idx]) + [EOS_IDX]
        trg = [SOS_IDX] + self.trg_vocab.encode(self.trg_sentences[idx]) + [EOS_IDX]
        return torch.LongTensor(src), torch.LongTensor(trg)


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """
    Assemble un batch de paires de longueurs variables en tenseurs paddés.
    Format time-first (seq_len, batch) — cohérent avec Encoder et Decoder.

    Returns:
        (src, src_lengths, trg, trg_lengths)
    """
    src_list, trg_list = zip(*batch)
    src_lengths = torch.tensor([len(s) for s in src_list])
    trg_lengths = torch.tensor([len(t) for t in trg_list])
    src = pad_sequence(src_list, batch_first=False, padding_value=PAD_IDX)
    trg = pad_sequence(trg_list, batch_first=False, padding_value=PAD_IDX)
    return src, src_lengths, trg, trg_lengths
