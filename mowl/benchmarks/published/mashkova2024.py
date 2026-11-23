"""Results published in Mashkova, Zhapa-Camacho, Hoehndorf, "DELE: Deductive EL++
Embeddings for Knowledge Base Completion" (arXiv:2411.01574v3), Tables 3-6.

Transcribed with ``pdftotext -layout`` (the tables are single-column and unambiguous).
Tables 3, 5, 6 columns: H@10 (NF, F), H@100 (NF, F), macro_MR (NF, F, NF-F), micro_MR
(NF, F, NF-F), macro_AUC (NF, F), micro_AUC (NF, F); the NF-F differences are not
recorded (they are derived). Table 4 columns: H@10, H@100, macro_MR, micro_MR,
macro_AUC, micro_AUC (non-filtered only).

Rows ``X+l`` (all negative losses) and ``X+l+n`` (all negative losses and negatives
filtered by the deductive closure) are the method ``X`` trained with the DELE training
regimes; ``settings["training_regime"]`` names the catalogued regimes. Hyperparameters
(embedding dimension, learning rate, margin) are from Appendix D; ``+l+n`` uses the
hyperparameters of ``+l`` (Section 6.3). All models: one fixed random seed, batch size
32,768, 2,000 epochs (GO & STRING) or 800 epochs (FoodOn, GALEN), ReduceLROnPlateau
(patience 10), early stopping after 20 epochs without validation improvement.
"""

from mowl.benchmarks.spec import PublishedResult
from mowl.benchmarks.entries._refs import MASHKOVA2024

_FILTERED = ("hits@10", "f_hits@10", "hits@100", "f_hits@100", "macro_mr", "f_macro_mr",
             None, "micro_mr", "f_micro_mr", None, "macro_auc", "f_macro_auc", "micro_auc",
             "f_micro_auc")
_RAW = ("hits@10", "hits@100", "macro_mr", "micro_mr", "macro_auc", "micro_auc")

_REGIMES = {
    "": (),
    "+l": ("dele-code-training",),
    "+l+n": ("dele-code-training",),
}

# Appendix D: (dim, learning rate, margin) per dataset and model; "+l+n" uses "+l".
_HYPER = {
    "yeast_ppi": {"ELEm": (100, 0.0001, -0.10), "ELEm+l": (50, 0.0001, 0.00),
                  "ELBE": (200, 0.0001, 0.00), "ELBE+l": (200, 0.0100, 0.00),
                  "Box2EL": (200, 0.0010, 0.01), "Box2EL+l": (200, 0.0010, 0.01)},
    "yeast_function": {"ELEm": (200, 0.0001, 0.01), "ELEm+l": (50, 0.0001, -0.10),
                       "ELBE": (200, 0.0001, 0.10), "ELBE+l": (200, 0.0001, 0.10),
                       "Box2EL": (200, 0.0100, 0.10), "Box2EL+l": (200, 0.0100, 0.10)},
    "foodon": {"ELEm": (400, 0.0010, -0.10), "ELEm+l": (400, 0.0010, -0.10),
               "ELBE": (200, 0.0100, 0.10), "ELBE+l": (200, 0.0100, -0.01),
               "Box2EL": (100, 0.0100, 0.10), "Box2EL+l": (200, 0.0010, 0.10)},
    "galen": {"ELEm": (400, 0.0010, -0.10), "ELEm+l": (400, 0.0010, -0.01),
              "ELBE": (100, 0.0010, 0.10), "ELBE+l": (200, 0.0010, 0.01),
              "Box2EL": (200, 0.0010, 0.00), "Box2EL+l": (200, 0.0100, 0.00)},
}

# Appendix D: ε of the "+l" rows (ELEm: not stated in the paper; 0.01 as in the
# reference code's run.py), δ and λ of the Box2EL rows (baseline and "+l").
_EPSILON = {
    "yeast_ppi": {"ELEm+l": 0.01, "ELBE+l": 0.001, "Box2EL+l": 0.010},
    "yeast_function": {"ELEm+l": 0.01, "ELBE+l": 0.010, "Box2EL+l": 0.010},
    "foodon": {"ELEm+l": 0.01, "ELBE+l": 0.001, "Box2EL+l": 0.010},
    "galen": {"ELEm+l": 0.01, "ELBE+l": 0.010, "Box2EL+l": 0.100},
}
_BOX2EL_EXTRA = {
    "yeast_ppi": {"Box2EL": (1, 0.05), "Box2EL+l": (2, 0.05)},
    "yeast_function": {"Box2EL": (4, 0.20), "Box2EL+l": (4, 0.05)},
    "foodon": {"Box2EL": (1, 0.20), "Box2EL+l": (4, 0.10)},
    "galen": {"Box2EL": (4, 0.05), "Box2EL+l": (1, 0.05)},
}

_TABLES = [
    ("dele-yeast-function", "yeast_function", "Table 3", _FILTERED, """
ELEm 0.01 0.01 0.03 0.03 21198 21150 48 21165 21118 47 0.62 0.62 0.63 0.63
ELEm+l 0.00 0.00 0.03 0.03 9603 9575 28 9449 9423 26 0.83 0.83 0.84 0.84
ELEm+l+n 0.00 0.00 0.03 0.03 9488 9460 28 9334 9307 27 0.83 0.83 0.84 0.84
ELBE 0.03 0.03 0.24 0.24 4229 4209 20 4156 4137 19 0.92 0.92 0.93 0.93
ELBE+l 0.00 0.00 0.01 0.01 12920 12865 55 12797 12745 52 0.77 0.77 0.78 0.78
ELBE+l+n 0.00 0.00 0.01 0.01 12900 12845 55 12772 12719 53 0.77 0.77 0.78 0.78
Box2EL 0.28 0.31 0.55 0.55 1988 1979 9 1988 1980 8 0.96 0.96 0.97 0.97
Box2EL+l 0.24 0.27 0.54 0.55 2129 2120 9 2099 2091 8 0.96 0.96 0.97 0.97
Box2EL+l+n 0.24 0.27 0.54 0.55 2161 2152 9 2147 2139 8 0.96 0.96 0.96 0.96"""),
    ("dele-yeast-ppi", "yeast_ppi", "Table 4", _RAW, """
ELEm 0.05 0.31 599.21 701.57 0.90 0.90
ELEm+l 0.06 0.35 532.93 681.02 0.91 0.90
ELEm+l+n 0.06 0.37 519.62 671.19 0.91 0.91
ELBE 0.07 0.37 829.86 1123.47 0.91 0.89
ELBE+l 0.08 0.40 984.92 1259.54 0.84 0.82
ELBE+l+n 0.08 0.40 984.18 1281.20 0.84 0.82
Box2EL 0.05 0.57 215.07 287.16 0.96 0.96
Box2EL+l 0.05 0.57 200.85 250.17 0.97 0.96
Box2EL+l+n 0.05 0.58 197.73 250.47 0.97 0.96"""),
    ("dele-foodon-subsumption", "foodon", "Table 5", _FILTERED, """
ELEm 0.12 0.12 0.21 0.21 4659 4656 3 4662 4659 3 0.84 0.84 0.84 0.84
ELEm+l 0.10 0.11 0.19 0.19 5015 5013 2 5020 5017 3 0.83 0.83 0.83 0.83
ELEm+l+n 0.10 0.11 0.19 0.19 5022 5019 3 5027 5024 3 0.83 0.83 0.83 0.83
ELBE 0.01 0.01 0.09 0.09 6695 6692 3 6688 6686 2 0.77 0.77 0.77 0.77
ELBE+l 0.04 0.04 0.14 0.14 5428 5426 2 5412 5409 3 0.81 0.81 0.82 0.82
ELBE+l+n 0.04 0.04 0.14 0.14 5427 5424 3 5410 5408 2 0.81 0.81 0.82 0.82
Box2EL 0.01 0.01 0.10 0.10 3900 3898 2 3877 3874 3 0.87 0.87 0.87 0.87
Box2EL+l 0.04 0.04 0.13 0.13 7550 7547 3 7555 7553 2 0.74 0.74 0.74 0.74
Box2EL+l+n 0.05 0.05 0.14 0.14 6865 6862 3 6869 6866 3 0.76 0.76 0.77 0.77"""),
    ("dele-galen-subsumption", "galen", "Table 6", _FILTERED, """
ELEm 0.15 0.16 0.35 0.35 9106 9105 1 9106 9105 1 0.82 0.82 0.82 0.82
ELEm+l 0.21 0.22 0.34 0.34 8977 8976 1 8977 8976 1 0.82 0.82 0.82 0.82
ELEm+l+n 0.21 0.22 0.33 0.34 9005 9003 2 9005 9003 2 0.82 0.82 0.82 0.82
ELBE 0.08 0.08 0.22 0.22 11236 11234 2 11236 11234 2 0.77 0.78 0.77 0.77
ELBE+l 0.13 0.13 0.34 0.34 11884 11882 2 11884 11882 2 0.76 0.76 0.76 0.76
ELBE+l+n 0.12 0.12 0.34 0.34 11720 11717 3 11720 11717 3 0.77 0.77 0.76 0.76
Box2EL 0.14 0.14 0.31 0.31 11724 11721 3 11724 11721 3 0.77 0.77 0.76 0.76
Box2EL+l 0.16 0.16 0.37 0.37 11371 11369 2 11371 11369 2 0.77 0.77 0.77 0.77
Box2EL+l+n 0.16 0.16 0.37 0.37 11378 11376 2 11378 11376 2 0.77 0.77 0.77 0.77"""),
]

_KEYS = {"ELEm": "elem", "ELBE": "elbe", "Box2EL": "box2el"}

RESULTS = []
for _bench, _data, _loc, _names, _text in _TABLES:
    for _line in _text.strip().splitlines():
        _parts = _line.split()
        _label = _parts[0]
        _base = _label.split("+")[0]
        _variant = _label[len(_base):]
        if len(_parts) - 1 != len(_names):
            raise ValueError(f"{_loc}: {_label} has {len(_parts) - 1} values")
        _metrics = {n: float(v) for n, v in zip(_names, _parts[1:]) if n is not None}
        _row = _base + ("+l" if _variant else "")
        _hp = _HYPER[_data][_row]
        _settings = {"embed_dim": _hp[0], "learning_rate": _hp[1], "margin": _hp[2]}
        if _variant:
            _settings["epsilon"] = _EPSILON[_data][_row]
        if _base == "Box2EL":
            _delta, _reg = _BOX2EL_EXTRA[_data][_row]
            _settings["delta"] = _delta
            _settings["reg_factor"] = _reg
        if _REGIMES[_variant]:
            _settings["training_regime"] = _REGIMES[_variant]
        RESULTS.append(PublishedResult(
            benchmark=_bench, method=_KEYS[_base], metrics=_metrics, source=MASHKOVA2024,
            locator=f"{_loc} ({_label})", settings=_settings,
            recipe=_variant or "base",
            note=(f"DELE variant '{_label}'." if _variant else
                  "Original negative sampling (GCI2 only).")))
