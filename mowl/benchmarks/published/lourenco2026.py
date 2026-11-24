"""Results published in Lourenço, Morgan, Ozaki, Pavlović, Sallinger, "BoxLitE: A Faithful
Knowledge Base Embedding Based on Convex Optimization" (KR 2026, arXiv:2605.23937v2),
Table 2: filtered MRR and Hits@{1,3,10} of the test role assertions of F_v1 - F_v4.

Transcribed from the rendered PDF page (checked against pdftotext). BoxLitE_i is the
ablation without the regulariser weighted by lambda_i (recorded as method "boxlite" with
lambda_i = 0 in ``settings``); the best lambdas are those of Tables 9-12 (Appendix G.3).
BoxE, RotatE and ComplEx were trained by the authors with PyKEEN 1.11.1 on the ABox only
(no TBox), d = 32, 100 trials of 500 epochs, Adagrad, early stopping, mean of 3 runs
(Appendix G.6). The F_v4 BoxLitE_2 row prints H@3 = ".54", which cannot be right
(H@1 = .023, H@10 = .110); that value is not recorded.
"""

from mowl.benchmarks.spec import PublishedResult
from mowl.benchmarks.entries.boxlite import LOURENCO2026

# rows: "model mrr h1 h3 h10" per dataset
_TABLE2 = {
    1: """BoxLitE1 .632 .435 .800 .962
BoxLitE2 .249 .134 .284 .475
BoxLitE3 .698 .524 .842 .984
BoxLitE .720 .545 .870 .979
BoxE .826 .719 .918 .986
RotatE .474 .295 .584 .805
ComplEx .322 .206 .359 .535""",
    2: """BoxLitE1 .428 .268 .519 .726
BoxLitE2 .144 .072 .142 .273
BoxLitE3 .541 .342 .668 .897
BoxLitE .549 .352 .685 .905
BoxE .432 .337 .461 .604
RotatE .209 .147 .218 .314
ComplEx .341 .265 .357 .488""",
    3: """BoxLitE1 .364 .239 .396 .626
BoxLitE2 .116 .055 .108 .224
BoxLitE3 .414 .280 .472 .681
BoxLitE .433 .288 .509 .708
BoxE .894 .827 .946 .998
RotatE .177 .104 .199 .298
ComplEx .184 .119 .205 .284""",
    4: """BoxLitE1 .339 .204 .415 .592
BoxLitE2 .059 .023 - .110
BoxLitE3 .409 .272 .483 .638
BoxLitE .444 .249 .571 .763
BoxE .626 .444 .772 .929
RotatE .245 .129 .300 .450
ComplEx .134 .087 .156 .203""",
}

# Best lambdas (lambda_1, lambda_2, lambda_3), Tables 9 (BoxLitE), 10-12 (ablations).
_LAMBDAS = {
    "BoxLitE": {1: (.001, 3, .001), 2: (.001, 1, .003), 3: (.003, .1, .003),
                4: (.001, 3, .003)},
    "BoxLitE1": {1: (0, 3, .001), 2: (0, 3, .03), 3: (0, .3, 0), 4: (0, .3, 0)},
    "BoxLitE2": {1: (.003, 0, 0), 2: (.01, 0, 1), 3: (.01, 0, .003), 4: (.3, 0, .003)},
    "BoxLitE3": {1: (.003, .1, 0), 2: (.003, 1, 0), 3: (.003, .1, 0), 4: (.003, .1, 0)},
}
_TABLES = {"BoxLitE": "Table 9", "BoxLitE1": "Table 10", "BoxLitE2": "Table 11",
           "BoxLitE3": "Table 12"}
_METHODS = {
    "BoxLitE": ("boxlite", "BoxLitE (MOSEK, reference code)."),
    "BoxLitE1": ("boxlite", "Ablation BoxLitE_1: without the negative concept "
                            "regulariser (lambda_1 = 0)."),
    "BoxLitE2": ("boxlite", "Ablation BoxLitE_2: without the head/tail width regulariser "
                            "(lambda_2 = 0)."),
    "BoxLitE3": ("boxlite", "Ablation BoxLitE_3: without the bump width regulariser "
                            "(lambda_3 = 0)."),
    "BoxE": ("boxe", "PyKEEN BoxE on the ABox only, mean of 3 runs."),
    "RotatE": ("rotate", "PyKEEN RotatE on the ABox only, mean of 3 runs."),
    "ComplEx": ("complex", "PyKEEN ComplEx on the ABox only, mean of 3 runs."),
}
_KEYS = ("mrr", "hits@1", "hits@3", "hits@10")

RESULTS = []
for _v, _text in _TABLE2.items():
    for _line in _text.splitlines():
        _name, *_values = _line.split()
        _method, _note = _METHODS[_name]
        _metrics = {k: float(x) for k, x in zip(_KEYS, _values) if x != "-"}
        _settings = {"embed_dim": 32}
        if _name.startswith("BoxLitE"):
            l1, l2, l3 = _LAMBDAS[_name][_v]
            _settings.update({"lambda_1": l1, "lambda_2": l2, "lambda_3": l3,
                              "s_omega": 1, "epsilon": 0.01, "solver": "MOSEK",
                              "lambdas_from": _TABLES[_name]})
        RESULTS.append(PublishedResult(
            benchmark=f"boxlite-family-v{_v}", method=_method, metrics=_metrics,
            source=LOURENCO2026, locator="Table 2", settings=_settings,
            note=_note + (" H@3 printed as '.54' omitted." if "-" in _values else "")))
