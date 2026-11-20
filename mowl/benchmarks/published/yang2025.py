"""Results published in Yang, Chen, Sattler, "TransBox: EL++-closed Ontology Embedding"
(WWW 2025): the normalised-axiom tables only (Tables 7, 8, 11, 12, 13).

TransBox's complex-axiom tables (Tables 2-6) are not recorded: their test sets are
generated with the forgetting tool LETHE (Appendix D) and are not part of any public
release that mOWL can load, and mOWL's EL models cannot score complex concepts in
general. Column order: H@1, H@10, H@100, Med, MRR, MR, AUC. Transcribed with pdftotext
and checked against the rendered pages.

Rows of ELEm and EmEL++ in Tables 11-13 are copied by the paper from Jackermeier et al.
(2024) ("We use the ELEM and ELEM++ results from [17]") and are identical to the rows
recorded in :mod:`mowl.benchmarks.published.jackermeier2024`; they are not recorded
again here. Box2EL, BoxEL, ELBE and CatE were re-run by the TransBox authors and differ
from the Box2EL paper.
"""

from mowl.benchmarks.spec import PublishedResult
from mowl.benchmarks.entries._refs import YANG2025

_KEYS = {"Box2EL": "box2el", "BoxEL": "boxel", "ELBE": "elbe", "TransBox": "transbox",
         "CatE": "cate"}
_NAMES = ("hits@1", "hits@10", "hits@100", "med", "mrr", "mr", "auc")

_NOTE = ("Average of 10 runs, 5,000 epochs, hyperparameters selected on validation "
         "(Section 6.2). Re-implementation of the Box2EL evaluation with a corrected "
         "score for C ⊓ D ⊑ E: the score is set to 0 when Box(C) ∩ Box(D) is empty "
         "(Appendix E).")

_TABLES = [
    ("galen", "combined", "Table 7 (GALEN)", """
Box2EL 0.04 0.17 0.31 1360 0.08 5183 0.78
BoxEL 0.00 0.03 0.16 4750 0.01 7213 0.69
ELBE 0.01 0.07 0.14 5340 0.03 7447 0.68
TransBox 0.01 0.08 0.18 4986 0.03 7227 0.69"""),
    ("go", "combined", "Table 7 (GO)", """
Box2EL 0.02 0.17 0.52 86 0.07 4593 0.90
BoxEL 0.01 0.06 0.08 8572 0.03 15116 0.67
ELBE 0.03 0.13 0.19 6836 0.06 10809 0.76
TransBox 0.01 0.16 0.35 1503 0.06 8199 0.82"""),
    ("anatomy", "combined", "Table 7 (Anatomy)", """
Box2EL 0.16 0.47 0.70 13 0.26 2954 0.97
BoxEL 0.03 0.11 0.25 1527 0.06 11930 0.89
ELBE 0.02 0.27 0.47 162 0.10 9562 0.91
TransBox 0.03 0.35 0.62 29 0.13 8186 0.92"""),
    ("galen", "gci1", "Table 8 (GALEN)", """
Box2EL 0.00 0.00 0.00 10910 0.00 11063 0.52
BoxEL 0.00 0.00 0.01 11217 0.00 11437 0.51
ELBE 0.00 0.00 0.00 10898 0.00 11053 0.52
TransBox 0.03 0.21 0.39 996 0.08 5112 0.78"""),
    ("go", "gci1", "Table 8 (GO)", """
Box2EL 0.00 0.00 0.01 14273 0.00 15566 0.66
BoxEL 0.00 0.00 0.00 22263 0.00 22569 0.51
ELBE 0.00 0.00 0.01 14712 0.00 16109 0.65
TransBox 0.06 0.60 0.77 7 0.21 2783 0.94"""),
    ("anatomy", "gci1", "Table 8 (Anatomy)", """
Box2EL 0.00 0.01 0.02 29327 0.00 34052 0.68
BoxEL 0.00 0.00 0.01 16982 0.00 24917 0.77
ELBE 0.01 0.02 0.06 22228 0.01 28524 0.73
TransBox 0.05 0.29 0.57 56 0.12 2215 0.98"""),
    ("galen", "gci0", "Table 11 (GALEN)", """
Box2EL 0.04 0.30 0.51 89 0.12 2648 0.89
CatE 0.00 0.13 0.42 - 0.06 2034 0.91
BoxEL 0.00 0.00 0.05 3427 0.00 5656 0.76
ELBE 0.02 0.12 0.28 751 0.06 3711 0.84
TransBox 0.00 0.03 0.11 6518 0.01 8120 0.65"""),
    ("go", "gci0", "Table 11 (GO)", """
Box2EL 0.03 0.17 0.57 58 0.08 2705 0.94
CatE 0.00 0.24 0.55 - 0.16 2521 0.94
BoxEL 0.00 0.01 0.04 5594 0.00 13734 0.70
ELBE 0.04 0.17 0.22 5098 0.08 9179 0.80
TransBox 0.01 0.09 0.24 3076 0.04 9137 0.80"""),
    ("anatomy", "gci0", "Table 11 (Anatomy)", """
Box2EL 0.07 0.34 0.65 27 0.15 2918 0.97
CatE 0.00 0.23 0.51 - 0.08 7939 0.92
BoxEL 0.01 0.04 0.13 2109 0.02 10036 0.91
ELBE 0.03 0.13 0.30 1353 0.06 11724 0.89
TransBox 0.06 0.25 0.50 100 0.12 11575 0.89"""),
    ("galen", "gci2", "Table 12 (GALEN)", """
Box2EL 0.08 0.18 0.32 662 0.12 3832 0.83
BoxEL 0.00 0.02 0.07 7638 0.01 8792 0.62
ELBE 0.00 0.07 0.13 5835 0.03 7623 0.67
TransBox 0.00 0.08 0.19 4115 0.03 6597 0.72"""),
    ("go", "gci2", "Table 12 (GO)", """
Box2EL 0.00 0.18 0.52 82 0.05 5085 0.89
BoxEL 0.00 0.00 0.00 16735 0.00 18848 0.59
ELBE 0.00 0.13 0.24 5092 0.03 11161 0.76
TransBox 0.00 0.24 0.53 63 0.06 6808 0.85"""),
    ("anatomy", "gci2", "Table 12 (Anatomy)", """
Box2EL 0.21 0.56 0.75 7 0.33 2457 0.98
BoxEL 0.04 0.17 0.33 885 0.08 12686 0.88
ELBE 0.01 0.36 0.59 33 0.13 7667 0.93
TransBox 0.02 0.42 0.70 17 0.14 6050 0.94"""),
    ("galen", "gci3", "Table 13 (GALEN)", """
Box2EL 0.00 0.07 0.16 4514 0.02 7317 0.68
BoxEL 0.00 0.14 0.69 49 0.04 2869 0.88
ELBE 0.00 0.00 0.01 11030 0.00 11139 0.52
TransBox 0.00 0.01 0.07 7665 0.00 8835 0.62"""),
    ("go", "gci3", "Table 13 (GO)", """
Box2EL 0.00 0.37 0.64 20 0.09 4971 0.89
BoxEL 0.08 0.55 0.55 3492 0.27 10293 0.78
ELBE 0.00 0.03 0.06 13218 0.01 15417 0.66
TransBox 0.00 0.07 0.34 2096 0.02 9905 0.78"""),
    ("anatomy", "gci3", "Table 13 (Anatomy)", """
Box2EL 0.00 0.05 0.15 2891 0.02 8284 0.92
BoxEL 0.00 0.00 0.00 25795 0.00 30281 0.72
ELBE 0.00 0.00 0.00 16239 0.00 26970 0.75
TransBox 0.00 0.01 0.12 3562 0.01 12631 0.88"""),
]

RESULTS = []
for _onto, _subset, _loc, _text in _TABLES:
    for _line in _text.strip().splitlines():
        _parts = _line.split()
        _metrics = {n: float(v) for n, v in zip(_NAMES, _parts[1:]) if v != "-"}
        _note = _NOTE
        if _parts[0] == "CatE":
            _note += " The median rank of CatE is not reported ('-')."
        RESULTS.append(PublishedResult(
            benchmark=f"transbox-{_onto}-prediction", method=_KEYS[_parts[0]],
            metrics=_metrics, source=YANG2025, locator=_loc, subset=_subset,
            settings={"training_regime": "transbox-code-training"}, note=_note))
