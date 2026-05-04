"""
Tests pour le dataset, le collate_fn et la loss.
Lancer : pytest tests/test_data_loss.py -v
"""

import torch, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from data import (
    Vocab, TranslationDataset, collate_fn,
    PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, SPECIAL_TOKENS,
)


# ─── Fixtures communes ───────────────────────────────────────────────────────
CORPUS_SRC = [
    ["je", "mange"],
    ["tu", "dors"],
    ["il", "mange", "une", "pomme"],
]
CORPUS_TRG = [
    ["I", "eat"],
    ["you", "sleep"],
    ["he", "eats", "an", "apple"],
]


# ─── Vocab ──────────────────────────────────────────────────────────────────
class TestVocab:
    def test_specials_first(self):
        v = Vocab.build(CORPUS_SRC)
        for i, tok in enumerate(SPECIAL_TOKENS):
            assert v.itos[i] == tok
            assert v.stoi[tok] == i

    def test_len(self):
        v = Vocab.build(CORPUS_SRC)
        # 4 specials + {je, mange, tu, dors, il, une, pomme} = 11
        assert len(v) == 11

    def test_encode_unknown(self):
        """Un token absent doit être mappé à UNK_IDX."""
        v = Vocab.build(CORPUS_SRC)
        assert v.encode(["inconnu"]) == [UNK_IDX]

    def test_encode_decode_roundtrip(self):
        v = Vocab.build(CORPUS_SRC)
        sentence = ["je", "mange"]
        assert v.decode(v.encode(sentence)) == sentence


# ─── Dataset ────────────────────────────────────────────────────────────────
class TestDataset:
    def setup_method(self):
        self.src_v = Vocab.build(CORPUS_SRC)
        self.trg_v = Vocab.build(CORPUS_TRG)
        self.ds = TranslationDataset(CORPUS_SRC, CORPUS_TRG, self.src_v, self.trg_v)

    def test_length(self):
        assert len(self.ds) == len(CORPUS_SRC)

    def test_src_ends_with_eos(self):
        src_ids, _ = self.ds[0]
        assert src_ids[-1].item() == EOS_IDX

    def test_trg_starts_sos_ends_eos(self):
        _, trg_ids = self.ds[0]
        assert trg_ids[0].item()  == SOS_IDX
        assert trg_ids[-1].item() == EOS_IDX


# ─── Collate ────────────────────────────────────────────────────────────────
class TestCollate:
    def setup_method(self):
        src_v = Vocab.build(CORPUS_SRC)
        trg_v = Vocab.build(CORPUS_TRG)
        ds    = TranslationDataset(CORPUS_SRC, CORPUS_TRG, src_v, trg_v)
        self.batch = [ds[i] for i in range(3)]

    def test_src_shape_time_first(self):
        out = collate_fn(self.batch)
        src = out["src"]
        assert src.dim() == 2
        assert src.shape[1] == 3   # batch

    def test_lengths_match_real_lengths(self):
        out = collate_fn(self.batch)
        expected = torch.tensor([len(self.batch[i][0]) for i in range(3)])
        assert torch.equal(out["src_lengths"], expected)

    def test_padding_value_is_pad_idx(self):
        """Les positions au-delà de la longueur réelle doivent être PAD_IDX."""
        out = collate_fn(self.batch)
        src, lens = out["src"], out["src_lengths"]
        for b in range(3):
            assert torch.all(src[lens[b]:, b] == PAD_IDX)


# ─── Loss ───────────────────────────────────────────────────────────────────
class TestLoss:
    def setup_method(self):
        from loss import Seq2SeqLoss
        self.loss = Seq2SeqLoss()

    def test_scalar_output(self):
        logits = torch.randn(5, 4, 20)
        target = torch.randint(1, 20, (5, 4))     # pas de PAD
        value = self.loss(logits, target)
        assert value.dim() == 0   # scalaire

    def test_pad_positions_ignored(self):
        """Changer les logits aux positions PAD ne doit pas changer la loss."""
        logits = torch.randn(5, 4, 20)
        target = torch.randint(1, 20, (5, 4))
        target[3:, :] = PAD_IDX                   # les 2 derniers pas sont du PAD
        loss_a = self.loss(logits, target)

        # On modifie uniquement les logits aux positions paddées
        logits_modified = logits.clone()
        logits_modified[3:, :, :] += 10.0
        loss_b = self.loss(logits_modified, target)

        assert torch.allclose(loss_a, loss_b, atol=1e-6), \
            "La loss ne doit pas dépendre des logits aux positions PAD"

    def test_backward_works(self):
        logits = torch.randn(5, 4, 20, requires_grad=True)
        target = torch.randint(1, 20, (5, 4))
        self.loss(logits, target).backward()
        assert logits.grad is not None
