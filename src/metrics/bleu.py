"""
BLEU — wrapper autour de sacrebleu (standard de l'industrie).

sacrebleu garantit une implémentation reproductible et correcte
(smoothing, tokenisation, edge cases) — référence : Post (2018).

Fonctions exposées :
    sentence_bleu(hypothesis, references) -> float
    corpus_bleu(hypotheses, references)   -> float
"""

import sacrebleu


def sentence_bleu(
    hypothesis : list[str],
    references : list[list[str]],
) -> float:
    """
    BLEU pour une seule phrase.

    Args:
        hypothesis : tokens de l'hypothèse         ex. ["le", "chat", "dort"]
        references : liste de références tokenisées ex. [["le", "chat", "dort", "."]]

    Returns:
        score BLEU entre 0 et 100
    """
    hyp  = " ".join(hypothesis)
    refs = [" ".join(r) for r in references]
    return sacrebleu.sentence_bleu(hyp, refs).score


def corpus_bleu(
    hypotheses      : list[list[str]],
    references_list : list[list[list[str]]],
) -> float:
    """
    BLEU au niveau corpus — agrège les comptes avant division (correct).

    Args:
        hypotheses      : liste d'hypothèses tokenisées
        references_list : liste de listes de références tokenisées
                          references_list[i] = références pour hypotheses[i]

    Returns:
        score BLEU entre 0 et 100
    """
    hyps = [" ".join(h) for h in hypotheses]
    # sacrebleu attend : refs[ref_idx][sent_idx]
    n_refs = len(references_list[0])
    refs   = [[" ".join(references_list[i][j]) for i in range(len(hypotheses))]
              for j in range(n_refs)]
    return sacrebleu.corpus_bleu(hyps, refs).score
