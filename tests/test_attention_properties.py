"""
Tests de propriétés mathématiques de la Scaled Dot-Product Attention.
Lancer : pytest tests/test_attention_properties.py -v
"""

import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

HIDDEN_DIM = 16
BATCH      = 4
SEQ_LEN    = 6


@pytest.fixture
def attn():
    from attention import ScaledDotProductAttention
    return ScaledDotProductAttention(d_model=HIDDEN_DIM, dropout=0.0)


def test_scaling_effect(attn):
    """Diviser par sqrt(d_k) doit rendre les scores moins piqués qu'avec d_k=1."""
    Q = K = V = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM) * 10   # grands scores
    _, alpha_scaled   = attn(Q, K, V)
    # Entropie doit être > 0 (pas une distribution de Dirac)
    entropy = -(alpha_scaled * (alpha_scaled + 1e-9).log()).sum(dim=-1)
    assert (entropy > 0).all(), "L'attention devrait être distribuée, pas concentrée"


def test_context_is_weighted_sum(attn):
    """context doit être exactement alpha @ V."""
    Q = torch.randn(BATCH, 3, HIDDEN_DIM)
    K = V = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    context, alpha = attn(Q, K, V)
    expected = torch.bmm(alpha, V)   # (batch, 3, hidden)
    assert torch.allclose(context, expected, atol=1e-5)


def test_different_queries_different_attention(attn):
    """Deux requêtes différentes → distributions d'attention différentes."""
    K = V = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    Q1 = torch.randn(BATCH, 1, HIDDEN_DIM)
    Q2 = torch.randn(BATCH, 1, HIDDEN_DIM)
    _, alpha1 = attn(Q1, K, V)
    _, alpha2 = attn(Q2, K, V)
    assert not torch.allclose(alpha1, alpha2)


def test_causal_mask_blocks_future(attn):
    """Avec masque causal, le token t ne doit voir que t' ≤ t."""
    from attention import causal_mask
    Q = K = V = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    mask = causal_mask(SEQ_LEN, torch.device("cpu"))
    mask = mask.unsqueeze(0).expand(BATCH, -1, -1)
    _, alpha = attn(Q, K, V, mask=mask)
    # Toutes les positions futures doivent avoir alpha ≈ 0
    for t in range(SEQ_LEN):
        future_weights = alpha[:, t, t+1:]
        assert torch.all(future_weights < 1e-6), \
            f"t={t} : voit des tokens futurs ({future_weights.max().item():.2e})"


def test_deterministic_in_eval(attn):
    attn.eval()
    Q = K = V = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    _, alpha1 = attn(Q, K, V)
    _, alpha2 = attn(Q, K, V)
    assert torch.allclose(alpha1, alpha2)
