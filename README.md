# Seq2Seq EN→FR — BiLSTM + Cross-Attention

Implémentation pédagogique d'un modèle de traduction anglais → français en PyTorch.

## Architecture

```
SOURCE  →  [BiLSTM Encoder]  →  enc_outputs (K, V)
                                       ↓
TARGET  →  [Embedding]  →  [LSTM 1]  →  [Cross-Attention]  →  [LSTM 2]  →  [Linear]  →  logits
```

- **Encoder** : BiLSTM — lit la source dans les deux sens, projette `hidden*2 → hidden`
- **LSTM 1** : lit les tokens cibles décalés (`<SOS>`, w1, w2, ...)
- **Cross-Attention** : chaque position cible consulte toute la séquence source (MHA de PyTorch)
- **LSTM 2** : intègre le contexte source pour produire la représentation finale
- **Padding** : `pack_padded_sequence` pour les LSTM, `key_padding_mask` pour le MHA

## Structure

```
seq2seq-attention/
├── src/
│   ├── attention.py     — CrossAttention (nn.MultiheadAttention)
│   ├── encoder.py       — BiLSTM Encoder
│   ├── decoder.py       — LSTM1 → Attention → LSTM2
│   ├── seq2seq.py       — Assemblage Encoder + Decoder
│   ├── data.py          — Vocab, TranslationDataset, collate_fn
│   ├── loss.py          — Seq2SeqLoss (CrossEntropy + ignore PAD)
│   ├── train.py         — train_epoch, evaluate
│   └── tests/
│       ├── test_shapes.py
│       ├── test_attention_properties.py
│       ├── test_gradients.py
│       └── test_data_loss.py
├── requirements.txt
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Tests

```bash
cd src
pytest tests/ -v
```

## Données

Utilise le dataset [Tatoeba](https://huggingface.co/datasets/tatoeba) via HuggingFace :

```python
from datasets import load_dataset
ds = load_dataset("tatoeba", lang1="en", lang2="fr")
```

## Évaluation

- **BLEU** via `sacrebleu`
- **BERTScore** via `bert-score`

Seuils indicatifs EN→FR sur Tatoeba : BLEU 15–25 pour ce type d'architecture.
