"""
Tests de gradient — vérifient que les poids reçoivent bien des gradients
après un backward pass (détecte les connexions coupées / detach() mal placés).

Lancer : pytest tests/test_gradients.py -v
"""

import torch
import torch.nn as nn
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

SRC_VOCAB = 20; TRG_VOCAB = 25; EMBED_DIM = 8
HIDDEN_DIM = 16; N_LAYERS = 1; BATCH = 4; SRC_LEN = 6; TRG_LEN = 5


def build_model():
    from encoder import Encoder
    from decoder import Decoder
    from seq2seq import Seq2Seq
    device = torch.device("cpu")
    enc = Encoder(SRC_VOCAB, EMBED_DIM, HIDDEN_DIM, N_LAYERS, 0.0)
    dec = Decoder(TRG_VOCAB, EMBED_DIM, HIDDEN_DIM, N_LAYERS, 0.0)
    return Seq2Seq(enc, dec, device)


class TestGradients:
    def setup_method(self):
        self.model = build_model()
        self.src   = torch.randint(0, SRC_VOCAB, (SRC_LEN, BATCH))
        self.trg   = torch.randint(0, TRG_VOCAB, (TRG_LEN, BATCH))

    def test_all_params_receive_gradient(self):
        """Tous les paramètres doivent avoir un gradient non-None après backward."""
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        outputs   = self.model(self.src, self.trg)

        # On calcule une loss bidon sur toute la séquence cible
        loss = criterion(
            outputs[1:].reshape(-1, TRG_VOCAB),
            self.trg[1:].reshape(-1),
        )
        loss.backward()

        params_without_grad = [
            name for name, p in self.model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not params_without_grad, \
            f"Paramètres sans gradient : {params_without_grad}"

    def test_attention_weights_grad(self):
        """Vérifie que W1, W2, v de l'attention reçoivent un gradient."""
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        outputs   = self.model(self.src, self.trg)
        loss = criterion(outputs[1:].reshape(-1, TRG_VOCAB), self.trg[1:].reshape(-1))
        loss.backward()

        for name in ["decoder.attention.W1", "decoder.attention.W2", "decoder.attention.v"]:
            param = dict(self.model.named_parameters())[name + ".weight"] \
                    if name + ".weight" in dict(self.model.named_parameters()) \
                    else dict(self.model.named_parameters()).get(name + ".weight",
                         next(p for n, p in self.model.named_parameters() if n.startswith(name)))
            assert param.grad is not None, f"{name} n'a pas de gradient"
