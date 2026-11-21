"""Published filtered link prediction results on FB15k-237 and WN18RR.

Transcribed from the LaTeX sources of the arXiv versions:

* Abboud et al. 2020 (arXiv:2007.06267), Table 3: BoxE with uniform (u) and
  self-adversarial (a) negative sampling;
* Pavlović and Sallinger 2023 (arXiv:2206.04192), Table 3: Base and Functional ExpressivE
  (and BoxE(a), copied from Abboud et al.).
"""

from mowl.benchmarks.spec import PublishedResult
from mowl.benchmarks.entries._refs import ABBOUD2020, PAVLOVIC2023

# (source, locator, method, note, settings, {dataset: "MR MRR H@1 H@3 H@10"}); "-" = not given
_ROWS = [
    (ABBOUD2020, "Table 3", "boxe", "BoxE(u): uniform negative sampling.",
     {"negative_sampling": "uniform"},
     {"fb15k237": "172 .318 .223 .351 .514", "wn18rr": "3117 .442 .398 .461 .523"}),
    (ABBOUD2020, "Table 3", "boxe", "BoxE(a): self-adversarial negative sampling.",
     {"negative_sampling": "self-adversarial"},
     {"fb15k237": "163 .337 .238 .374 .538", "wn18rr": "3207 .451 .400 .472 .541"}),
    (PAVLOVIC2023, "Table 3", "expressive", "Base ExpressivE.", {"variant": "base"},
     {"fb15k237": "- .333 .243 .366 .512", "wn18rr": "- .508 .464 .522 .597"}),
    (PAVLOVIC2023, "Table 3", "expressive",
     "Functional ExpressivE (width parameters fixed to 0); mOWL implements the base model.",
     {"variant": "functional"},
     {"fb15k237": "- .350 .256 .387 .535", "wn18rr": "- .482 .407 .519 .619"}),
]
_NAMES = ("f_mr", "f_mrr", "f_hits@1", "f_hits@3", "f_hits@10")

RESULTS = []
for _source, _locator, _method, _note, _settings, _by_data in _ROWS:
    for _data, _text in _by_data.items():
        _metrics = {n: float(v) for n, v in zip(_NAMES, _text.split()) if v != "-"}
        RESULTS.append(PublishedResult(
            benchmark=f"kg-{_data}-link-prediction", method=_method, metrics=_metrics,
            source=_source, locator=_locator, settings=_settings, note=_note))
