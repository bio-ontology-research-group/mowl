"""Results published in Jackermeier, Chen, Horrocks, "Dual Box Embeddings for the
Description Logic EL++" (WWW 2024), Tables 1-6.

Transcribed from the PDF (pdftotext -layout, checked against the rendered pages). Column
order of Tables 1-4 and 6: H@1, H@10, H@100, Med, MRR, MR, AUC. Table 5: H@10, H@10 (F),
H@100, H@100 (F), MR, MR (F), AUC, AUC (F). The prediction and deductive reasoning
numbers are means over 5 seeds (Section 4.2).
"""

from mowl.benchmarks.spec import PublishedResult
from mowl.benchmarks.entries._refs import JACKERMEIER2024

_KEYS = {"ELEm": "elem", "EmEL++": "emelpp", "BoxEL": "boxel", "ELBE": "elbe",
         "Box2EL": "box2el"}
_RANK = ("hits@1", "hits@10", "hits@100", "med", "mrr", "mr", "auc")
_PPI = ("hits@10", "f_hits@10", "hits@100", "f_hits@100", "mr", "f_mr", "auc", "f_auc")

# (benchmark, subset, locator, rows "Model v1 ... vn")
_TABLES = [
    ("box2el-galen-prediction", "combined", "Table 1 (GALEN)", """
ELEm 0.01 0.12 0.29 1662 0.05 5153 0.78
EmEL++ 0.01 0.11 0.24 2295 0.05 5486 0.76
BoxEL 0.00 0.03 0.16 4785 0.01 7163 0.69
ELBE 0.02 0.14 0.27 1865 0.06 5303 0.77
Box2EL 0.05 0.20 0.35 669 0.10 4375 0.81"""),
    ("box2el-go-prediction", "combined", "Table 1 (GO)", """
ELEm 0.03 0.24 0.43 272 0.09 6204 0.86
EmEL++ 0.03 0.23 0.38 597 0.09 6710 0.85
BoxEL 0.01 0.06 0.08 8443 0.03 14905 0.68
ELBE 0.01 0.10 0.22 1838 0.04 6986 0.85
Box2EL 0.04 0.23 0.59 48 0.10 3248 0.93"""),
    ("box2el-anatomy-prediction", "combined", "Table 1 (Anatomy)", """
ELEm 0.10 0.40 0.64 22 0.19 6464 0.94
EmEL++ 0.11 0.36 0.57 36 0.19 8472 0.92
BoxEL 0.03 0.12 0.28 1151 0.06 10916 0.90
ELBE 0.04 0.36 0.63 29 0.15 5400 0.95
Box2EL 0.16 0.47 0.70 13 0.26 2675 0.97"""),
    ("box2el-galen-prediction", "gci0", "Table 2 (NF1)", """
ELEm 0.01 0.16 0.40 430 0.06 3568 0.85
EmEL++ 0.02 0.16 0.37 632 0.06 3765 0.84
BoxEL 0.00 0.00 0.05 3715 0.00 5727 0.75
ELBE 0.03 0.24 0.47 138 0.10 2444 0.89
Box2EL 0.03 0.30 0.51 91 0.12 2632 0.89"""),
    ("box2el-galen-prediction", "gci1", "Table 2 (NF2)", """
ELEm 0.01 0.07 0.17 5106 0.03 7432 0.68
EmEL++ 0.01 0.07 0.15 5750 0.03 7767 0.66
BoxEL 0.00 0.00 0.00 11358 0.00 11605 0.50
ELBE 0.03 0.06 0.11 6476 0.04 8068 0.65
Box2EL 0.06 0.15 0.28 2149 0.09 6265 0.73"""),
    ("box2el-galen-prediction", "gci2", "Table 2 (NF3)", """
ELEm 0.02 0.14 0.28 1479 0.05 4831 0.79
EmEL++ 0.02 0.11 0.22 2240 0.05 5348 0.77
BoxEL 0.00 0.02 0.08 7239 0.01 8615 0.63
ELBE 0.03 0.14 0.25 2154 0.07 5072 0.78
Box2EL 0.08 0.19 0.32 635 0.12 3798 0.84"""),
    ("box2el-galen-prediction", "gci3", "Table 2 (NF4)", """
ELEm 0.00 0.05 0.18 3855 0.02 6793 0.71
EmEL++ 0.00 0.04 0.12 4458 0.01 7020 0.70
BoxEL 0.00 0.15 0.69 47 0.04 2667 0.89
ELBE 0.00 0.03 0.07 7563 0.01 8884 0.62
Box2EL 0.00 0.06 0.15 4364 0.02 7266 0.69"""),
    ("box2el-go-prediction", "gci0", "Table 3 (NF1)", """
ELEm 0.01 0.13 0.35 590 0.05 6433 0.86
EmEL++ 0.01 0.12 0.30 1023 0.05 6709 0.85
BoxEL 0.00 0.01 0.05 5374 0.00 13413 0.71
ELBE 0.01 0.10 0.24 1156 0.04 5657 0.88
Box2EL 0.03 0.17 0.58 58 0.08 2686 0.94"""),
    ("box2el-go-prediction", "gci1", "Table 3 (NF2)", """
ELEm 0.12 0.49 0.63 11 0.24 4508 0.90
EmEL++ 0.11 0.44 0.55 23 0.21 5169 0.89
BoxEL 0.00 0.00 0.00 22882 0.00 23007 0.50
ELBE 0.01 0.05 0.09 6456 0.02 9421 0.80
Box2EL 0.18 0.58 0.75 6 0.31 2104 0.95"""),
    ("box2el-go-prediction", "gci2", "Table 3 (NF3)", """
ELEm 0.06 0.40 0.52 54 0.15 6292 0.86
EmEL++ 0.05 0.39 0.48 210 0.15 7788 0.83
BoxEL 0.00 0.00 0.00 17027 0.00 18947 0.59
ELBE 0.02 0.15 0.30 959 0.07 7131 0.84
Box2EL 0.00 0.18 0.53 79 0.05 5042 0.89"""),
    ("box2el-go-prediction", "gci3", "Table 3 (NF4)", """
ELEm 0.01 0.49 0.60 12 0.12 6272 0.86
EmEL++ 0.01 0.49 0.58 12 0.13 6442 0.86
BoxEL 0.09 0.54 0.54 2215 0.28 9673 0.79
ELBE 0.00 0.07 0.12 9049 0.02 12868 0.72
Box2EL 0.00 0.37 0.64 20 0.08 4989 0.89"""),
    ("box2el-anatomy-prediction", "gci0", "Table 4 (NF1)", """
ELEm 0.07 0.30 0.57 43 0.14 9059 0.91
EmEL++ 0.08 0.29 0.53 60 0.14 10414 0.90
BoxEL 0.01 0.05 0.16 1828 0.03 9597 0.91
ELBE 0.05 0.24 0.55 68 0.11 5177 0.95
Box2EL 0.07 0.34 0.65 27 0.15 2894 0.97"""),
    ("box2el-anatomy-prediction", "gci1", "Table 4 (NF2)", """
ELEm 0.03 0.18 0.42 394 0.08 11592 0.89
EmEL++ 0.03 0.18 0.35 1291 0.08 15759 0.85
BoxEL 0.00 0.00 0.00 17607 0.00 26872 0.75
ELBE 0.02 0.11 0.26 1394 0.05 4885 0.96
Box2EL 0.16 0.41 0.64 26 0.24 1928 0.98"""),
    ("box2el-anatomy-prediction", "gci2", "Table 4 (NF3)", """
ELEm 0.12 0.47 0.69 13 0.23 4686 0.96
EmEL++ 0.13 0.42 0.60 23 0.23 7097 0.93
BoxEL 0.04 0.17 0.36 567 0.08 11095 0.90
ELBE 0.04 0.44 0.70 16 0.18 5408 0.95
Box2EL 0.21 0.56 0.75 7 0.33 2466 0.98"""),
    ("box2el-anatomy-prediction", "gci3", "Table 4 (NF4)", """
ELEm 0.00 0.03 0.23 813 0.01 10230 0.91
EmEL++ 0.00 0.02 0.17 1470 0.01 10951 0.90
BoxEL 0.00 0.00 0.00 38942 0.00 41283 0.61
ELBE 0.00 0.02 0.06 6261 0.01 15187 0.86
Box2EL 0.00 0.05 0.14 3065 0.01 8366 0.92"""),
    ("box2el-galen-inference", "all", "Table 6 (GALEN)", """
ELEm 0.00 0.04 0.20 1807 0.01 4405 0.81
EmEL++ 0.00 0.04 0.18 2049 0.01 4634 0.81
BoxEL 0.00 0.00 0.01 6906 0.00 7925 0.67
ELBE 0.00 0.06 0.16 1785 0.02 3974 0.84
Box2EL 0.01 0.09 0.24 1003 0.03 2833 0.88"""),
    ("box2el-go-inference", "all", "Table 6 (GO)", """
ELEm 0.00 0.04 0.22 1629 0.02 7377 0.84
EmEL++ 0.00 0.04 0.19 1346 0.01 6557 0.86
BoxEL 0.00 0.00 0.13 1085 0.00 5359 0.88
ELBE 0.00 0.06 0.21 935 0.02 3846 0.92
Box2EL 0.00 0.08 0.49 107 0.04 1689 0.96"""),
    ("box2el-anatomy-inference", "all", "Table 6 (Anatomy)", """
ELEm 0.00 0.07 0.28 901 0.02 7958 0.93
EmEL++ 0.00 0.07 0.26 1576 0.02 10976 0.90
BoxEL 0.01 0.10 0.24 838 0.04 9156 0.92
ELBE 0.00 0.08 0.32 336 0.03 2312 0.98
Box2EL 0.01 0.09 0.44 152 0.04 1599 0.99"""),
]

_PPI_TABLES = [
    ("box2el-yeast-ppi", "Table 5 (Yeast)", """
ELEm 0.10 0.23 0.50 0.75 247 187 0.96 0.97
EmEL++ 0.08 0.17 0.48 0.65 336 291 0.94 0.95
BoxEL 0.09 0.20 0.52 0.73 423 379 0.93 0.94
ELBE 0.11 0.26 0.57 0.77 201 154 0.96 0.97
Box2EL 0.11 0.33 0.64 0.87 168 118 0.97 0.98"""),
    ("box2el-human-ppi", "Table 5 (Human)", """
ELEm 0.09 0.22 0.43 0.70 658 572 0.96 0.96
EmEL++ 0.04 0.13 0.38 0.56 772 700 0.95 0.95
BoxEL 0.07 0.10 0.42 0.63 1574 1530 0.93 0.93
ELBE 0.09 0.22 0.49 0.72 434 362 0.97 0.98
Box2EL 0.09 0.28 0.55 0.83 343 269 0.98 0.98"""),
]

_PPI_NOTE = {
    "boxel": "Copied by the paper from Xiong et al. (2022), as stated in the caption of "
             "Table 5.",
    "box2el": "",
}
_PPI_DEFAULT_NOTE = ("Copied by the paper from Peng et al. (ELBE paper), as stated in the "
                     "caption of Table 5; not re-run under the reference evaluation code.")


def _rows(text, names):
    for line in text.strip().splitlines():
        parts = line.split()
        yield _KEYS[parts[0]], dict(zip(names, (float(v) for v in parts[1:])))


_BOXEL_NF4_NOTE = (
    " Not reproducible: the reference evaluation of BoxEL for ∃r.C ⊑ D "
    "(evaluate.py, compute_nf4_ranks_boxel) ranks the class whose index equals the role "
    "index instead of D (see the regime 'overlap-volume-scoring'); the combined numbers "
    "include these ranks.")


def _note(method, subset, bench):
    note = ("Mean over 5 seeds; baselines re-run by the authors with the Box2EL "
            "evaluation code.")
    if (method == "boxel" and bench.endswith("-prediction")
            and subset in ("gci3", "combined")):
        note += _BOXEL_NF4_NOTE
    return note


RESULTS = []
for _bench, _subset, _loc, _text in _TABLES:
    for _method, _metrics in _rows(_text, _RANK):
        RESULTS.append(PublishedResult(
            benchmark=_bench, method=_method, metrics=_metrics, source=JACKERMEIER2024,
            locator=_loc, subset=_subset,
            settings={"training_regime": "box2el-code-training"},
            note=_note(_method, _subset, _bench)))
for _bench, _loc, _text in _PPI_TABLES:
    for _method, _metrics in _rows(_text, _PPI):
        RESULTS.append(PublishedResult(
            benchmark=_bench, method=_method, metrics=_metrics, source=JACKERMEIER2024,
            locator=_loc, note=_PPI_NOTE.get(_method, _PPI_DEFAULT_NOTE)))
