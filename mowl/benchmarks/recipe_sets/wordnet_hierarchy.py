"""Settings for the WordNet noun hierarchy benchmark (``wordnet-noun-closure-*``).

The publications train with stochastic (Riemannian) optimisers on mini-batches, some from
pretrained Poincaré embeddings with a burn-in phase. mOWL trains these models with its
generic full-batch loop from a random start, so the published optimiser settings do not
carry over; the settings here are mOWL's, written for dimension 10 (for 5 run with
``--param embed_dim=5``). Each recipe states where its values come from.
"""

from mowl.benchmarks.recipes import Recipe
from mowl.catalog.card import Reference

MOWL_WORDNET = Reference(
    "mowl-wordnet",
    "mOWL settings for the WordNet noun hierarchy benchmark (documentation, section "
    "'Reproducing published results').",
    "https://mowl.readthedocs.io/en/latest/benchmarks/index.html")

_NOTE = ("Full-batch training with mOWL's generic loop and 10 uniform corruptions of either "
         "side per positive, from a random start: not the optimiser of the publication.")

#: method -> (name, model arguments, epochs)
_SETTINGS = {
    "order_embeddings": [("", {"margin": 1.0, "learning_rate": 0.01}, 2000)],
    "box_lattice": [("", {"learning_rate": 0.01}, 2000)],
    "gumbel_box": [("", {"learning_rate": 0.01}, 2000)],
    "hyperbolic_cones": [("", {"K": 0.1, "margin": 0.01, "learning_rate": 0.1}, 2000)],
    "disk_embeddings": [
        ("hyperbolic", {"geometry": "hyperbolic", "margin": 0.01, "learning_rate": 0.1}, 2000),
        ("spherical", {"geometry": "spherical", "margin": 0.01, "learning_rate": 0.1}, 2000),
        ("euclidean", {"geometry": "euclidean", "margin": 0.01, "learning_rate": 0.1}, 2000)],
    "shadow_cones": [
        ("umbral-half-space", {"formulation": "umbral-half-space", "learning_rate": 0.01},
         2000),
        ("penumbral-half-space", {"formulation": "penumbral-half-space",
                                  "learning_rate": 0.01}, 2000)],
}


def _build():
    out = []
    for percent in (0, 10, 25, 50, 90):
        for method, variants in _SETTINGS.items():
            for name, kwargs, epochs in variants:
                out.append(Recipe(
                    benchmark=f"wordnet-noun-closure-{percent}", method=method,
                    source=MOWL_WORDNET, name=name,
                    model_kwargs={"embed_dim": 10, **kwargs}, epochs=epochs,
                    locator="mowl.benchmarks.recipe_sets.wordnet_hierarchy", notes=_NOTE))
    return tuple(out)


RECIPES = _build()
