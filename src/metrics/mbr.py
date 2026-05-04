"""
MBR — Minimum Bayes Risk decoding (Kumar & Byrne, 2004).

Idée :
    Au lieu de choisir l'hypothèse de plus grande probabilité (MAP / beam search),
    on choisit celle qui MINIMISE LE RISQUE BAYÉSIEN attendu — autrement dit,
    celle qui maximise l'utilité espérée face aux autres hypothèses plausibles.

    Formulation :
        ŷ = argmax_{h ∈ H}  E_{y ~ p(y|x)} [ utility(h, y) ]

    Approximation Monte-Carlo (on n'a pas accès à p(y|x) directement) :
        - On échantillonne un ensemble de candidats H = {h_1, ..., h_N}   (via sampling)
        - On utilise ce même ensemble comme pseudo-référence pour estimer l'espérance
        - ŷ = argmax_{h_i}  (1/N) * sum_{j}  utility(h_i, h_j)

    Utility typiques :
        - BLEU, ROUGE, METEOR, chrF, BERTScore, ...
        - Toute fonction utility(candidate, reference) -> float  (plus grand = mieux)

Pourquoi c'est intéressant :
    - Plus robuste au model bias que le MAP pur
    - Tire parti de la redondance entre bons candidats (un bon candidat ressemble
      aux autres bons candidats)
"""

from typing import Callable, Sequence


Tokens        = Sequence[str]
UtilityFn     = Callable[[Tokens, Tokens], float]


def mbr_decode(
    candidates: Sequence[Tokens],
    utility   : UtilityFn,
) -> tuple[int, Tokens]:
    """
    Sélectionne le candidat MBR parmi un pool de candidats échantillonnés.

    Args:
        candidates : liste d'hypothèses (issues de sampling, beam, etc.)
        utility    : fonction utility(h, ref) -> float
                     ex. lambda h, r: sentence_bleu(h, [r])

    Returns:
        (best_index, best_candidate)
    """
    # TODO:
    # 1. N = len(candidates)
    # 2. Pour chaque i dans 0..N-1 :
    #       score_i = (1 / N) * sum_{j != i}  utility(candidates[i], candidates[j])
    #    (ou inclure j=i si utility(h,h) ≈ 1 — choix de design à documenter)
    # 3. best_i = argmax_i score_i
    # 4. return best_i, candidates[best_i]
    raise NotImplementedError


def mbr_decode_matrix(
    candidates: Sequence[Tokens],
    utility   : UtilityFn,
) -> tuple[int, Tokens, list[list[float]]]:
    """
    Variante qui retourne aussi la matrice d'utilités (pour debug / visualisation).

    Returns:
        (best_index, best_candidate, utility_matrix)
        utility_matrix[i][j] = utility(candidates[i], candidates[j])
    """
    # TODO: même logique, mais stocker U[i][j] puis score_i = mean(U[i, :])
    raise NotImplementedError
