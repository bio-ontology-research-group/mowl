"""Recipes of DELE (Mashkova, Zhapa-Camacho, Hoehndorf, arXiv:2411.01574, Tables 3-6):
one recipe per (benchmark, method, DELE variant), trained the way the reference
implementation (bio-ontology-research-group/DELE, commit 5974c7d) trained.

Every recipe uses the ``dele-code-training`` regime, which bundles the reference's loss
functions (several of which differ from mOWL's and the papers' models; see the
``dele-code-losses`` regime card) and how its training loops combine them
(``dele-code-loss``), fixed negatives (``negative_seed=0``: the reference resets NumPy's
seed to 0 before drawing the negatives of every batch, so that every positive keeps one
negative for the whole training; runs with negatives drawn anew moved several cells
away from the published numbers, e.g. ELBE on yeast PPI), cyclic mini-batches of 32,768 axioms per normal form,
ReduceLROnPlateau (patience 10) and early stopping (patience 20) on the validation loss
of the predicted normal form, monitored once per paper epoch. Model arguments are from
Appendix D of the paper; the "+l+n" rows reuse the "+l" hyperparameters (Section 6.3).

The reference counts an epoch as :math:`\\max_t \\lceil n_t / b \\rceil` steps of
:math:`b` = 32,768 axioms per normal form. With the GO & STRING data that is 9 steps
(GCI2 has 269,567 / 290,433 axioms), so ``epochs`` below is the paper's 2,000 epochs
times 9; FoodOn and GALEN have at most 29,618 axioms per normal form, one step covers
everything, and 800 epochs are 800 steps. ``validate_every`` is the same number of steps
(one validation per paper epoch).

The paper does not state :math:`\\varepsilon` for ELEmbeddings ("we do not tune the
parameter ε", Section 6.3); the recipes use 0.01, the value of the reference code's
example (``run.py``). On the GO & STRING data the reference corrupts a class with a
class of the same kind (yeast protein vs the rest); the recipes pass
``kind_prefixes=('4932.',)``.
"""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries._refs import MASHKOVA2024

#: Paper epochs and steps per paper epoch, per benchmark (Appendix A-C: the largest
#: normal form has 269,567 / 290,433 GCI2 axioms for the yeast data, at most 29,618
#: axioms per normal form for FoodOn and GALEN).
_EPOCHS = {"dele-yeast-function": 2000, "dele-yeast-ppi": 2000,
           "dele-foodon-subsumption": 800, "dele-galen-subsumption": 800}
_STEPS = {"dele-yeast-function": 9, "dele-yeast-ppi": 9,
          "dele-foodon-subsumption": 1, "dele-galen-subsumption": 1}

#: Protein IRIs of the GO & STRING data (the per-kind corruption pools of the reference
#: dataloader are split by this prefix).
_KIND = {"dele-yeast-function": ("4932.",), "dele-yeast-ppi": ("4932.",)}

#: Appendix D: benchmark -> variant -> method -> model arguments. "+l+n" uses "+l".
_HP = {
    "dele-yeast-function": {
        "": {"elem": {"embed_dim": 200, "learning_rate": 0.0001, "margin": 0.01},
             "elbe": {"embed_dim": 200, "learning_rate": 0.0001, "margin": 0.10},
             "box2el": {"embed_dim": 200, "learning_rate": 0.0100, "margin": 0.10,
                        "delta": 4, "reg_factor": 0.20}},
        "+l": {"elem": {"embed_dim": 50, "learning_rate": 0.0001, "margin": -0.10},
               "elbe": {"embed_dim": 200, "learning_rate": 0.0001, "margin": 0.10},
               "box2el": {"embed_dim": 200, "learning_rate": 0.0100, "margin": 0.10,
                          "delta": 4, "reg_factor": 0.05}},
    },
    "dele-yeast-ppi": {
        "": {"elem": {"embed_dim": 100, "learning_rate": 0.0001, "margin": -0.10},
             "elbe": {"embed_dim": 200, "learning_rate": 0.0001, "margin": 0.00},
             "box2el": {"embed_dim": 200, "learning_rate": 0.0010, "margin": 0.01,
                        "delta": 1, "reg_factor": 0.05}},
        "+l": {"elem": {"embed_dim": 50, "learning_rate": 0.0001, "margin": 0.00},
               "elbe": {"embed_dim": 200, "learning_rate": 0.0100, "margin": 0.00},
               "box2el": {"embed_dim": 200, "learning_rate": 0.0010, "margin": 0.01,
                          "delta": 2, "reg_factor": 0.05}},
    },
    "dele-foodon-subsumption": {
        "": {"elem": {"embed_dim": 400, "learning_rate": 0.0010, "margin": -0.10},
             "elbe": {"embed_dim": 200, "learning_rate": 0.0100, "margin": 0.10},
             "box2el": {"embed_dim": 100, "learning_rate": 0.0100, "margin": 0.10,
                        "delta": 1, "reg_factor": 0.20}},
        "+l": {"elem": {"embed_dim": 400, "learning_rate": 0.0010, "margin": -0.10},
               "elbe": {"embed_dim": 200, "learning_rate": 0.0100, "margin": -0.01},
               "box2el": {"embed_dim": 200, "learning_rate": 0.0010, "margin": 0.10,
                          "delta": 4, "reg_factor": 0.10}},
    },
    "dele-galen-subsumption": {
        "": {"elem": {"embed_dim": 400, "learning_rate": 0.0010, "margin": -0.10},
             "elbe": {"embed_dim": 100, "learning_rate": 0.0010, "margin": 0.10},
             "box2el": {"embed_dim": 200, "learning_rate": 0.0010, "margin": 0.00,
                        "delta": 4, "reg_factor": 0.05}},
        "+l": {"elem": {"embed_dim": 400, "learning_rate": 0.0010, "margin": -0.01},
               "elbe": {"embed_dim": 200, "learning_rate": 0.0010, "margin": 0.01},
               "box2el": {"embed_dim": 200, "learning_rate": 0.0100, "margin": 0.00,
                          "delta": 1, "reg_factor": 0.05}},
    },
}

#: Appendix D: ε of the "+l" rows, per benchmark and method (ELEmbeddings: not stated in
#: the paper; 0.01 as in the reference code's run.py).
_EPSILON = {
    "dele-yeast-function": {"elem": 0.01, "elbe": 0.010, "box2el": 0.010},
    "dele-yeast-ppi": {"elem": 0.01, "elbe": 0.001, "box2el": 0.010},
    "dele-foodon-subsumption": {"elem": 0.01, "elbe": 0.001, "box2el": 0.010},
    "dele-galen-subsumption": {"elem": 0.01, "elbe": 0.010, "box2el": 0.100},
}

_LOCATOR = ("Appendix D of the paper (Table 3-6 hyperparameters); run.py and models/*.py "
            "of bio-ontology-research-group/DELE (commit 5974c7d)")

_VARIANT_NOTES = {
    "": ("Original negative sampling (GCI2 only) and the original loss functions of the "
         "model, as in the baseline rows of the tables."),
    "+l": ("All negative losses (DELE, Section 4): negatives for every normal form, "
           "scored with the DELE negative losses."),
    "+ln": ("All negative losses and negatives filtered by the deductive closure "
             "(DELE, Section 5); hyperparameters of the '+l' row, as in Section 6.3."),
}


def _build():
    out = []
    for benchmark, epochs in _EPOCHS.items():
        steps = _STEPS[benchmark]
        kind = _KIND.get(benchmark)
        for variant in ("", "+l", "+ln"):
            base = "+l" if variant == "+ln" else variant
            label = {"": "base", "+l": "+l", "+ln": "+l+n"}[variant]
            for method, model_kwargs in _HP[benchmark][base].items():
                regime_kwargs = {"negatives": "all" if base == "+l" else "gci2",
                                 "negative_seed": 0}
                if base == "+l":
                    regime_kwargs["epsilon"] = _EPSILON[benchmark][method]
                if variant == "+ln":
                    regime_kwargs["filter_negatives"] = True
                if kind is not None:
                    regime_kwargs["kind_prefixes"] = list(kind)
                out.append(Recipe(
                    benchmark=benchmark, method=method, source=MASHKOVA2024,
                    name=label, model_kwargs=dict(model_kwargs),
                    training_regime="dele-code-training", regime_kwargs=regime_kwargs,
                    epochs=epochs * steps, validate_every=steps,
                    locator=_LOCATOR,
                    notes=(f"DELE variant '{benchmark} {method}"
                           f"{'+' + label if label != 'base' else ''}'. "
                           + _VARIANT_NOTES[variant]
                           + (" Epsilon is not stated in the paper for ELEmbeddings; "
                              "0.01 as in the reference code's run.py."
                              if method == "elem" and base == "+l" else "")
                           + (" The epochs are the paper's "
                              f"{epochs} epochs times {steps} steps per epoch; "
                              "validation (and the learning-rate schedule) once per "
                              "paper epoch." if steps > 1 else ""))))
    return tuple(out)


RECIPES = _build()
