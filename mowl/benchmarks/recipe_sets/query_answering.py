"""Recipes for the logical query answering benchmarks (``query-*``).

Training of every recipe: **the released training queries** (``queries="released"``), not
queries sampled from the graph by mOWL's own sampler. All four papers train on them (the
codebases of BetaE, ConE and AConE share one loop, KGReasoning ``main.py``; see
:class:`mowl.models.composed.query.QueryEmbeddingModel`). One mOWL epoch is 1000 steps of that
loop, so ``epochs=450`` is ``--max_steps 450001``. The learning rate is divided by 5 at half
and at three quarters of the steps (the loop's ``warm_up_steps`` rule), the regime
``step-lr-decay``.

Sources of the values (the authors' code is followed where it differs from the paper, and
the difference is in ``notes``):

* BetaE and Query2Box on the BetaE data: ``example.sh`` of
  https://github.com/snap-stanford/KGReasoning (the paper's Appendix, Table 'Hyperparameters
  used for each method', gives the learning rate 5e-4, the margin 30 (Query2Box) and 60 (BetaE) and
  300k iterations);
* ConE: ``scripts.sh`` of https://github.com/MIRALab-USTC/QE-ConE and the paper's Table of
  hyperparameters;
* AConE: ``script.sh`` of https://github.com/RoyaHe/AConE and the paper's supplementary
  Table of hyperparameters.
"""

import math

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries._refs import (HE2024ACONE, REN2020BETAE, REN2020Q2B,
                                           ZHANG2021CONE)

_COMMON = {"queries": "released", "hierarchy_closure": False, "extension": False,
           "batch_size": 512, "negatives": 128, "steps_per_epoch": 1000}
_NOTE = ("mOWL trains on the released training queries, as the paper. Differences: the "
         "entities are classes and the training graph is read from the ontology (the "
         "inverse edges are added by the model); the model is evaluated after the last step "
         "(the code also evaluates every few steps and the paper selects on validation "
         "queries, which mOWL does not do here); the initialisation and the random "
         "streams differ; the geometry is mOWL's reimplementation (see the method card). "
         "An epoch is 1000 steps of the reference loop.")
_FIVE = ["1p", "2p", "3p", "2i", "3i"]


def _decay(epochs):
    return ("step-lr-decay", {"milestones": [epochs // 2, (3 * epochs) // 4], "gamma": 0.2})


def _recipe(benchmark, method, source, model_kwargs, epochs, locator, notes=""):
    regime, kwargs = _decay(epochs)
    return Recipe(benchmark=benchmark, method=method, source=source,
                  model_kwargs={**_COMMON, **model_kwargs}, training_regime=regime,
                  regime_kwargs=kwargs, epochs=epochs, locator=locator,
                  notes=(notes + " " + _NOTE).strip())


_KGR = "example.sh of KGReasoning (code), Appendix 'Experimental Details' of the paper"
_BETAE_PAPER = ("The paper (Appendix) states learning rate 5e-4, 300k iterations, margin 30 "
                "for Query2Box and 60 for BetaE, projection MLP of three layers with 512 "
                "latent dimensions; the code (followed) uses 1e-4, 450001 steps and an MLP of "
                "two layers with 1600 units.")

RECIPES = []
for _kg in ("fb15k", "fb15k237", "nell995"):
    _bench = f"query-{_kg}-betae"
    RECIPES.append(_recipe(
        _bench, "betae", REN2020BETAE,
        {"embed_dim": 400, "gamma": 60.0, "learning_rate": 1e-4, "hidden": 1600, "layers": 2},
        450, _KGR, _BETAE_PAPER + " Trained on the ten structures of the release (positive "
        "and negation)."))
    RECIPES.append(_recipe(
        _bench, "query2box", REN2020BETAE,
        {"embed_dim": 400, "gamma": 24.0, "learning_rate": 1e-4, "center_weight": 0.02,
         "activation": "none", "structures": _FIVE}, 450, _KGR,
        _BETAE_PAPER + " Query2Box is trained on 1p, 2p, 3p, 2i, 3i only (no negation)."))

# ConE: scripts.sh of QE-ConE; the paper's hyperparameter table lists the same values except
# the steps and the learning rate of FB15k-237 (see the notes).
_CONE = {
    "fb15k": dict(gamma=30.0, learning_rate=5e-5, dropout=0.05, epochs=450),
    "fb15k237": dict(gamma=30.0, learning_rate=5e-5, dropout=0.1, epochs=300),
    "nell995": dict(gamma=20.0, learning_rate=1e-4, dropout=0.2, epochs=300),
}
_CONE_NOTE = ("The paper's table of hyperparameters (Appendix B.2) has a misaligned header "
              "(the column of the margin holds the number of steps); read with the code, "
              "it gives 300k, 300k and 450k steps for FB15k, FB15k-237 and NELL where "
              "scripts.sh has 450001, 300001 and 300001, and the learning rate 1e-4 for "
              "FB15k-237 where scripts.sh has 5e-5; the code is followed. The paper's "
              "batch size 128 and negatives 512 are swapped relative to the code "
              "(-b 512 -n 128), which is followed. ConE's dropout is applied in the "
              "intersection (CardMin).")
for _kg, _p in _CONE.items():
    _epochs = _p.pop("epochs")
    RECIPES.append(_recipe(
        f"query-{_kg}-betae", "cone_qe", ZHANG2021CONE,
        {"embed_dim": 800, "center_weight": 0.02, "hidden": 1600, "layers": 2, **_p},
        _epochs, "scripts.sh of QE-ConE; Appendix B.2, Table 'Hyperparameters found by grid "
        "search'", _CONE_NOTE))

_ACONE = {"embed_dim": 800, "gamma": 20.0, "learning_rate": 1e-4, "center_weight": 0.02,
          "dropout": 0.2, "signed_negation": True}
_ACONE_NOTE = ("script.sh of the AConE code: --max_steps 300001, --drop 0.2, -cenr 0.02; "
               "the supplementary table gives d = 800, b = 512, n = 128, gamma = 20, "
               "l = 1e-4, lambda = 0.02. AConE's projection is the rotation (no MLP), so no "
               "hidden size is set. The paper's table does not give the dropout or the "
               "number of steps; the code's are used. The negation is the code's pi - aperture "
               "of the wrapped, possibly negative aperture (signed_negation=True).")
RECIPES.append(_recipe("query-nell995-betae", "acone", HE2024ACONE, _ACONE, 300,
                       "script.sh of AConE; supplementary Table of hyperparameters",
                       _ACONE_NOTE))
RECIPES.append(_recipe("query-wn18rr-qa", "acone", HE2024ACONE, _ACONE, 300,
                       "script.sh of AConE; supplementary Table of hyperparameters",
                       _ACONE_NOTE + " Needs the WN18RR-QA data, which mOWL does not fetch "
                       "(dataset_kwargs data_dir)."))

# Query2Box and BetaE on the Query2Box queries. Query2Box paper, Appendix 'Hyper-parameters':
# d = 400, gamma = 24, alpha = 0.2, minibatch 512 per structure, 128 negatives, Adam lr 1e-4,
# 250 epochs. Their code (TensorFlow) was not checked.
_N1P = {"fb15k": 273710, "fb15k237": 149689, "nell995": 107982}
for _kg in ("fb15k", "fb15k237", "nell995"):
    _steps = math.ceil(_N1P[_kg] / 512)
    _r = _recipe(
        f"query-{_kg}-q2b", "query2box", REN2020Q2B,
        {"embed_dim": 400, "gamma": 24.0, "learning_rate": 1e-4, "center_weight": 0.2,
         "structures": _FIVE, "steps_per_epoch": _steps}, 250,
        "Appendix 'Hyper-parameters' of the paper",
        "The paper trains 250 epochs; an epoch is read here as a pass over the 1p training "
        f"queries ({_N1P[_kg]} / 512 = {_steps} steps), which the paper does not define. "
        "The paper draws 512 queries of every structure per iteration; mOWL's released "
        "training mode draws 512 from the pool of path structures and 512 from the pool of "
        "the others, in the proportions of the release (the loop of the BetaE code), and "
        "uses its learning rate rule (divided by 5 at half and three quarters of the "
        "training), which the paper does not mention. alpha = 0.2 is the paper's; the "
        "KGReasoning code uses 0.02. The offset activation is mOWL's default (relu).")
    RECIPES.append(_r)
    RECIPES.append(_recipe(
        f"query-{_kg}-q2b", "betae", REN2020BETAE,
        {"embed_dim": 400, "gamma": 60.0, "learning_rate": 1e-4, "hidden": 1600, "layers": 2},
        450, _KGR, _BETAE_PAPER + " The same training queries as on the BetaE benchmark; "
        "tested on the Query2Box queries."))
