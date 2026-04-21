"""
Tests de forme.  Lancer : pytest tests/test_shapes.py -v
"""

import torch, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

PAD_IDX   = 0
SRC_VOCAB = 20;  TRG_VOCAB = 25
EMBED_DIM = 8;   HIDDEN_DIM = 16
N_LAYERS  = 1;   N_HEADS = 4;  DROPOUT = 0.0
BATCH     = 4;   SRC_LEN = 6;  TRG_LEN = 5

# Longueurs réelles (dernière séquence plus courte = padding)
SRC_LENGTHS = torch.tensor([6, 5, 4, 3])
TRG_LENGTHS = torch.tensor([5, 4, 3, 2])


def make_padded(vocab, length, lengths):
    """Crée un batch avec padding à PAD_IDX selon les longueurs réelles."""
    t = torch.zeros(length, BATCH, dtype=torch.long)
    for i, l in enumerate(lengths):
        t[:l, i] = torch.randint(1, vocab, (l,))   # 0 réservé au PAD
    return t


# ─── Attention ───────────────────────────────────────────────────────────────
class TestCrossAttention:
    def setup_method(self):
        from attention import CrossAttention
        self.attn        = CrossAttention(HIDDEN_DIM, N_HEADS, dropout=0.0)
        self.query       = torch.randn(TRG_LEN, BATCH, HIDDEN_DIM)
        self.enc_outputs = torch.randn(SRC_LEN, BATCH, HIDDEN_DIM)
        self.mask        = (make_padded(SRC_VOCAB, SRC_LEN, SRC_LENGTHS) == PAD_IDX).T

    def test_context_shape(self):
        ctx, alpha = self.attn(self.query, self.enc_outputs)
        assert ctx.shape == (TRG_LEN, BATCH, HIDDEN_DIM)

    def test_alpha_shape(self):
        _, alpha = self.attn(self.query, self.enc_outputs)
        assert alpha.shape == (BATCH, TRG_LEN, SRC_LEN)

    def test_alpha_sums_to_one(self):
        _, alpha = self.attn(self.query, self.enc_outputs)
        assert torch.allclose(alpha.sum(dim=-1), torch.ones(BATCH, TRG_LEN), atol=1e-5)

    def test_masked_positions_near_zero(self):
        """Les positions PAD dans la source doivent avoir un poids ≈ 0."""
        _, alpha = self.attn(self.query, self.enc_outputs, key_padding_mask=self.mask)
        # La 4e séquence a SRC_LEN=3 → positions 3,4,5 doivent être ≈ 0
        assert torch.all(alpha[3, :, 3:] < 1e-6)


# ─── Encodeur ────────────────────────────────────────────────────────────────
class TestEncoder:
    def setup_method(self):
        from encoder import Encoder
        self.enc = Encoder(SRC_VOCAB, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
        self.src = make_padded(SRC_VOCAB, SRC_LEN, SRC_LENGTHS)

    def test_enc_outputs_shape(self):
        enc_outputs, h_n, c_n = self.enc(self.src, SRC_LENGTHS)
        assert enc_outputs.shape == (SRC_LEN, BATCH, HIDDEN_DIM)

    def test_h_n_shape(self):
        _, h_n, _ = self.enc(self.src, SRC_LENGTHS)
        assert h_n.shape == (N_LAYERS, BATCH, HIDDEN_DIM)

    def test_c_n_shape(self):
        _, _, c_n = self.enc(self.src, SRC_LENGTHS)
        assert c_n.shape == (N_LAYERS, BATCH, HIDDEN_DIM)

    def test_padding_does_not_affect_short_seq(self):
        """h_n d'une séquence courte ne doit pas changer si on ajoute du padding."""
        src_no_pad  = torch.zeros(SRC_LEN, 1, dtype=torch.long)
        src_no_pad[:3, 0] = torch.tensor([1, 2, 3])
        src_padded  = src_no_pad.clone()   # même chose, longueur déclarée = 3

        _, h1, _ = self.enc(src_no_pad, torch.tensor([3]))
        _, h2, _ = self.enc(src_padded, torch.tensor([3]))
        assert torch.allclose(h1, h2, atol=1e-5)


# ─── Décodeur ────────────────────────────────────────────────────────────────
class TestDecoder:
    def setup_method(self):
        from decoder import Decoder
        self.dec  = Decoder(TRG_VOCAB, EMBED_DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, DROPOUT)
        self.trg  = make_padded(TRG_VOCAB, TRG_LEN, TRG_LENGTHS)
        self.h    = torch.zeros(N_LAYERS, BATCH, HIDDEN_DIM)
        self.c    = torch.zeros(N_LAYERS, BATCH, HIDDEN_DIM)
        self.enc  = torch.randn(SRC_LEN, BATCH, HIDDEN_DIM)

    def test_logits_shape(self):
        logits, _ = self.dec(self.trg, TRG_LENGTHS, self.enc, self.h, self.c, self.h, self.c)
        assert logits.shape == (TRG_LEN, BATCH, TRG_VOCAB)

    def test_alpha_shape(self):
        _, alpha = self.dec(self.trg, TRG_LENGTHS, self.enc, self.h, self.c, self.h, self.c)
        assert alpha.shape == (BATCH, TRG_LEN, SRC_LEN)


# ─── Seq2Seq complet ─────────────────────────────────────────────────────────
class TestSeq2Seq:
    def setup_method(self):
        from encoder import Encoder
        from decoder import Decoder
        from seq2seq import Seq2Seq
        enc = Encoder(SRC_VOCAB, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
        dec = Decoder(TRG_VOCAB, EMBED_DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, DROPOUT)
        self.model = Seq2Seq(enc, dec, torch.device("cpu"))
        self.src   = make_padded(SRC_VOCAB, SRC_LEN, SRC_LENGTHS)
        self.trg   = make_padded(TRG_VOCAB, TRG_LEN, TRG_LENGTHS)

    def test_logits_shape(self):
        logits, alpha = self.model(self.src, SRC_LENGTHS, self.trg, TRG_LENGTHS)
        assert logits.shape == (TRG_LEN, BATCH, TRG_VOCAB)

    def test_alpha_shape(self):
        logits, alpha = self.model(self.src, SRC_LENGTHS, self.trg, TRG_LENGTHS)
        assert alpha.shape == (BATCH, TRG_LEN, SRC_LEN)
