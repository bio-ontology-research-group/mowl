"""Published F1 on the WordNet noun hierarchy benchmark of Ganea et al. (2018).

Transcribed from the LaTeX sources of the arXiv versions:

* Ganea et al. 2018 (arXiv:1804.01882), Table 1: order embeddings and hyperbolic cones;
* Suzuki et al. 2019 (arXiv:1902.04335), Table 2 ("WordNet nouns"): disk embeddings, and
  their own runs of hyperbolic cones and order embeddings;
* Yu et al. 2024 (arXiv:2305.15215), Table 2 ("Noun"): shadow cones and their run of
  the entailment cones;
* Dasgupta et al. 2020 (arXiv:2010.04831), Table 3: Gumbel boxes (and order embeddings and
  hyperbolic cones copied from Ganea et al.), 0% setting, without regularisation.

F1 is printed in %; it is stored as a fraction. Rows for methods mOWL does not implement
(Poincaré embeddings, simple Euclidean embeddings, Euclidean cones, SmoothBox, GaussianBox)
are not recorded. Dasgupta et al. do not state the dimension of their WordNet runs in the
main text; the order embedding and cone numbers they quote are Ganea's d = 10 numbers.
"""

from mowl.benchmarks.spec import PublishedResult
from mowl.benchmarks.entries._refs import DASGUPTA2020, GANEA2018, SUZUKI2019, YU2023SHADOW

_PERCENTS = (0, 10, 25, 50)

# (source, locator, method, settings, note, {dim: "0% 10% 25% 50%"})
_ROWS = [
    (GANEA2018, "Table 1", "order_embeddings", {}, "Order Emb (the authors' run).",
     {5: "34.4 70.2 75.9 81.7", 10: "43.0 69.7 79.4 84.1"}),
    (GANEA2018, "Table 1", "hyperbolic_cones", {}, "Our Hyperbolic Cones.",
     {5: "29.2 80.1 86.0 92.8", 10: "32.2 85.9 91.0 94.4"}),
    (SUZUKI2019, "Table 2, WordNet nouns", "disk_embeddings", {"geometry": "euclidean"},
     "Our Euclidean Disk Embeddings.",
     {5: "35.6 38.9 42.5 45.1", 10: "45.6 54.0 65.8 72.0"}),
    (SUZUKI2019, "Table 2, WordNet nouns", "disk_embeddings", {"geometry": "hyperbolic"},
     "Our Hyperbolic Disk Embeddings.",
     {5: "32.9 69.1 81.3 83.1", 10: "36.5 79.7 90.5 94.2"}),
    (SUZUKI2019, "Table 2, WordNet nouns", "disk_embeddings", {"geometry": "spherical"},
     "Our Spherical Disk Embeddings.",
     {5: "37.5 84.8 90.5 93.4", 10: "42.0 86.4 91.5 93.9"}),
    (SUZUKI2019, "Table 2, WordNet nouns", "hyperbolic_cones", {},
     "Hyperbolic Entailment Cones as run by Suzuki et al.",
     {5: "29.2 80.0 87.1 92.8", 10: "32.4 84.9 90.8 93.8"}),
    (SUZUKI2019, "Table 2, WordNet nouns", "order_embeddings", {},
     "Order Embeddings as run by Suzuki et al.",
     {5: "34.4 70.6 75.9 82.1", 10: "43.0 69.7 79.4 84.1"}),
    (YU2023SHADOW, "Table 2, Noun", "hyperbolic_cones", {},
     "Entailment Cone as run by Yu et al.",
     {5: "29.2 78.1 84.6 92.1", 10: "32.1 82.9 91.0 95.2"}),
    (YU2023SHADOW, "Table 2, Noun", "shadow_cones", {"formulation": "umbral-half-space"},
     "Umbral-half-space.", {5: "45.2 87.8 94.2 96.4", 10: "52.2 89.4 95.7 97.0"}),
    (YU2023SHADOW, "Table 2, Noun", "shadow_cones", {"formulation": "penumbral-half-space"},
     "Penumbral-half-space.", {5: "44.6 82.6 86.2 88.3", 10: "51.7 84.1 88.3 89.8"}),
]

RESULTS = []
for _source, _locator, _method, _settings, _note, _by_dim in _ROWS:
    for _dim, _text in _by_dim.items():
        for _percent, _value in zip(_PERCENTS, _text.split()):
            RESULTS.append(PublishedResult(
                benchmark=f"wordnet-noun-closure-{_percent}", method=_method,
                metrics={"f1": round(float(_value) / 100, 4)}, source=_source,
                locator=_locator, settings={"embed_dim": _dim, **_settings}, note=_note))

RESULTS.append(PublishedResult(
    benchmark="wordnet-noun-closure-0", method="gumbel_box", metrics={"f1": 0.512},
    source=DASGUPTA2020, locator="Table 3, without regularization", settings={},
    note="GumbelBox, trained on the transitive reduction; 62.6 with the volume "
         "regularisation of Patel et al., which mOWL's max_regularization approximates."))
