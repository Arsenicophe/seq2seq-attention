"""
ROUGE — wrapper autour de rouge-score (Google).

Fonctions exposées :
    rouge_n(hypothesis, reference, n)        -> RougeScore
    corpus_rouge(hypotheses, references, n)  -> RougeScore (moyenne)
"""

from dataclasses import dataclass
from rouge_score import rouge_scorer as _rouge_scorer


@dataclass
class RougeScore:
    precision : float
    recall    : float
    f1        : float


def rouge_n(
    hypothesis : list[str],
    reference  : list[str],
    n          : int = 1,
) -> RougeScore:
    """
    ROUGE-N entre une hypothèse et une référence.

    Args:
        hypothesis : tokens de l'hypothèse
        reference  : tokens de la référence
        n          : ordre du n-gramme (1 ou 2)

    Returns:
        RougeScore(precision, recall, f1)
    """
    scorer = _rouge_scorer.RougeScorer([f"rouge{n}"], use_stemmer=False)
    result = scorer.score(" ".join(reference), " ".join(hypothesis))
    score  = result[f"rouge{n}"]
    return RougeScore(
        precision = score.precision,
        recall    = score.recall,
        f1        = score.fmeasure,
    )


def corpus_rouge(
    hypotheses : list[list[str]],
    references : list[list[str]],
    n          : int = 1,
) -> RougeScore:
    """
    ROUGE-N moyen sur un corpus.

    Args:
        hypotheses : liste d'hypothèses tokenisées
        references : liste de références tokenisées

    Returns:
        RougeScore moyen sur toutes les paires
    """
    scores = [rouge_n(h, r, n) for h, r in zip(hypotheses, references)]
    return RougeScore(
        precision = sum(s.precision for s in scores) / len(scores),
        recall    = sum(s.recall    for s in scores) / len(scores),
        f1        = sum(s.f1        for s in scores) / len(scores),
    )
