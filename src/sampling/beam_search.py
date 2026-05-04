"""
Beam Search — décodage par recherche en faisceau.

Idée :
    À chaque pas de temps, on garde les `beam_size` meilleures hypothèses partielles
    (selon leur log-proba cumulée), au lieu de se limiter à la meilleure (greedy)
    ou d'explorer tout l'arbre (intractable).

Algorithme (haut niveau) :
    1. Init : beams = [ ([<SOS>], log_prob=0.0, state=h0) ]
    2. À chaque pas t :
        a. Pour chaque beam vivant, calculer p(y_t | y_<t, x) via le decodeur
        b. Candidats = beam × vocab  → garder top `beam_size` par log-proba cumulée
        c. Les beams qui génèrent <EOS> passent dans `finished`
    3. Stop quand `beam_size` beams finished ou max_len atteint
    4. Length normalization : score_final = log_prob / len^alpha    (α ∈ [0.6, 1.0])
       — sinon le beam favorise les séquences courtes

Pourquoi on garde l'état LSTM par beam :
    Chaque beam est une séquence différente → l'état caché (h, c) du decodeur
    diverge. On ne peut pas factoriser un seul forward pass comme à l'entraînement.

Architecture :
    - On ne consulte l'Encoder qu'UNE FOIS (enc_outputs constants pour toute la
      phrase source) → on les passe en entrée à chaque step de décodage.
    - Le decodeur doit exposer un mode step-by-step qui retourne (h, c).
"""

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F

from seq2seq import Seq2Seq


# Indices des tokens spéciaux (cf. data.py)
PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2


@dataclass
class Beam:
    tokens   : List[int]                # séquence de tokens générés (incluant <SOS>)
    log_prob : float                    # log-proba cumulée
    h        : torch.Tensor | None      # état caché LSTM1
    c        : torch.Tensor | None      # cell state LSTM1
    finished : bool = False             # True si <EOS> a été émis

    def score(self, alpha: float = 0.7) -> float:
        """Score normalisé par longueur — évite le biais vers séquences courtes."""
        return self.log_prob / (len(self.tokens) ** alpha)


@torch.no_grad()
def beam_search_decode(
    model     : Seq2Seq,
    src       : torch.Tensor,           # (src_len, 1)  — une seule phrase
    src_length: torch.Tensor,           # (1,)          — longueur réelle
    beam_size : int   = 5,
    max_len   : int   = 50,
    alpha     : float = 0.7,
    device    : torch.device = torch.device("cpu"),
) -> List[int]:
    """
    Décodage beam search pour UNE phrase source.

    Args:
        model      : Seq2Seq entraîné (mode eval)
        src        : tokens source (src_len, 1)
        src_length : longueur réelle de la source
        beam_size  : largeur du faisceau
        max_len    : longueur max de la séquence cible
        alpha      : exposant pour la length normalization

    Returns:
        Liste des indices de tokens du meilleur candidat (sans <SOS>, avec <EOS>
        si présent).
    """
    model.eval()
    model.decoder.return_state = True

    # 1. Encoder — passe unique, enc_outputs constants pour tous les beams
    enc_outputs          = model.encoder(src, src_length)             # (src_len, 1, H)
    src_key_padding_mask = (src == PAD_IDX).T.to(device)             # (1, src_len)

    # 2. Init — un seul beam au départ
    beams: List[Beam]    = [Beam(tokens=[SOS_IDX], log_prob=0.0, h=None, c=None)]
    finished: List[Beam] = []

    # 3. Boucle sur les pas de temps
    for t in range(1, max_len + 1):

        candidates: List[Beam] = []

        for beam in beams:

            # a. Entrée décodeur : dernier token du beam courant
            trg         = torch.tensor([[beam.tokens[-1]]], device=device)  # (1, 1)
            trg_lengths = torch.tensor([1])

            # b. Appel décodeur — récupère logits + nouveaux états LSTM1
            logits, _, (h_new, c_new) = model.decoder(
                trg, trg_lengths, enc_outputs,
                beam.h, beam.c,
                src_key_padding_mask=src_key_padding_mask,
            )                                                         # logits : (1, 1, vocab)

            # c. Log-probabilités sur le vocab
            log_probs = F.log_softmax(logits[-1, 0], dim=-1)         # (vocab_size,)

            # d. Créer un candidat pour chaque token possible
            for token_id in range(log_probs.size(0)):
                new_score = beam.log_prob + log_probs[token_id].item()
                candidates.append(Beam(
                    tokens   = beam.tokens + [token_id],
                    log_prob = new_score,
                    h        = h_new,
                    c        = c_new,
                ))

        # e. Garder les beam_size meilleurs candidats globaux
        candidates.sort(key=lambda b: b.log_prob, reverse=True)
        top_candidates = candidates[:beam_size]

        # f. Séparer finished / actifs
        beams = []
        for candidate in top_candidates:
            if candidate.tokens[-1] == EOS_IDX:
                finished.append(candidate)
            else:
                beams.append(candidate)

        # g. Stop si assez de séquences terminées ou plus de beams actifs
        if len(finished) >= beam_size or not beams:
            break

    # 4. Choisir le meilleur dans finished (fallback : meilleur beam actif)
    pool = finished if finished else beams
    best = max(pool, key=lambda b: b.score(alpha))

    # 5. Retourner la séquence sans le <SOS> initial
    return best.tokens[1:]
