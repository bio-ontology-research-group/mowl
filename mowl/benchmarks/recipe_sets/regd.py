"""Recipes of RegD (Yang and Chen, arXiv:2501.17518) and its ELBE / ELEm baselines, from
the paper and the authors' code (HuiYang1997/RegD, EXP_ontology).

What the authors' code gives. The released sweep (``run_all_sweeps.py``: learning rate
0.0005 / 0.005 / 0.01 times margin 0 / 0.05 / 0.1 / 0.15, 5000 steps, d = 10) was run as
released on the inference task, after two repairs without which it does not start (the
keyword ``depth_type=`` of the model constructors, which take ``plugin_type=``, and two
undefined names in ``model/loaded_models.py``). The best TEST F1 of the twelve
configurations, next to the paper:

=========  ====================  ==========  =====
Ontology   Model (plugin)        Code, best  Paper
=========  ====================  ==========  =====
ANATOMY    ELBE (tBox)           0.43        0.43
ANATOMY    ELEm (tBox)           0.39        0.39
ANATOMY    RegD boxes            0.39        0.63
ANATOMY    RegD balls            0.54        0.56
GO         ELBE (tBox)           0.40        0.42
GO         ELEm (tBox)           0.28        0.27
GO         RegD boxes            0.36        0.51
GO         RegD balls            0.40        0.40
=========  ====================  ==========  =====

So the published numbers of the baselines and of the balls are the maximum over the test
scores of the sweep, and the released code does not give the published numbers of RegD
with boxes (the paper's weight lambda and its p are not in the released code). The
"ELBE" and "ELEm" of the paper are the code's models with its ``tBox`` plugin, which
score lower than mOWL's ELBE and ELEm under the same training (0.69 to 0.76 on these two
tasks); mOWL's RegD with boxes (0.41 and 0.49 with the configuration chosen on the
validation set) is at or above what the released code reaches.
"""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries._refs import YANG2025_REGD

_LOC = ("EXP_ontology of HuiYang1997/RegD: baseline_sweep.py (5000 steps, batch 512, "
        "validation every 100), model/pluginRegD.py, model/Elbe.py, model/Elem.py and "
        "run_all_sweeps.py (the sweep)")

_NOTE = (
    "The published numbers are the best of a learning-rate x margin sweep "
    "(lr 0.0005/0.005/0.01, margin 0/0.05/0.1/0.15) per dataset, task, model and "
    "dimension; the code writes the TEST F1 of every configuration to one results file "
    "and nothing in the code selects among them by validation, so the published value is "
    "at best a maximum over 12 test scores. This recipe fixes one configuration, chosen "
    "by VALIDATION F1 in a learning-rate sweep at d = 10, seed 0 (see the reproduction "
    "report). Written for d = 10; for d = 5 run with --param embed_dim=5. The released "
    "baseline_sweep.py passes depth_type= to Elbe/Elem, whose constructors take "
    "plugin_type=, so the released script cannot run as published; mOWL follows the "
    "evident intent (plugin_type = the sweep's value).")

_ONTO = {"galen": "GALEN", "go": "GO", "anatomy": "ANATOMY"}

#: Learning rate per (method, ontology, task): the best VALIDATION F1 of the code's
#: learning rates {0.0005, 0.005, 0.01} at d = 10, seed 0 (runs regd2-w1-*; the ELEm
#: ANATOMY prediction run at 0.01 failed, 0.005 is used).
_LR = {
    ("regd", "galen", "inference"): 0.01, ("regd", "galen", "prediction"): 0.005,
    ("regd", "go", "inference"): 0.0005, ("regd", "go", "prediction"): 0.005,
    ("regd", "anatomy", "inference"): 0.01, ("regd", "anatomy", "prediction"): 0.0005,
    ("regd_elem", "galen", "inference"): 0.0005, ("regd_elem", "galen", "prediction"): 0.01,
    ("regd_elem", "go", "inference"): 0.005, ("regd_elem", "go", "prediction"): 0.01,
    ("regd_elem", "anatomy", "inference"): 0.005,
    ("regd_elem", "anatomy", "prediction"): 0.01,
    ("elbe", "galen", "inference"): 0.005, ("elbe", "galen", "prediction"): 0.01,
    ("elbe", "go", "inference"): 0.01, ("elbe", "go", "prediction"): 0.005,
    ("elbe", "anatomy", "inference"): 0.01, ("elbe", "anatomy", "prediction"): 0.01,
    ("elem", "galen", "inference"): 0.01, ("elem", "galen", "prediction"): 0.01,
    ("elem", "go", "inference"): 0.01, ("elem", "go", "prediction"): 0.01,
    ("elem", "anatomy", "inference"): 0.01, ("elem", "anatomy", "prediction"): 0.005,
}


#: Weight lambda of the depth term per (method, task), from the paper's grid {0, 0.5, 1}
#: (appendix C.1; the released code has no lambda, i.e. 1), chosen by VALIDATION F1 at
#: d = 10, seed 0 (runs regd2-w1, regd2-w3-*-lamda*, regd2-w5-*). For boxes on the
#: inference task lambda = 0 (no depth term) has the best validation F1 on GO and
#: ANATOMY (and ties on GALEN); everywhere else lambda = 1.
_LAMDA = {
    ("regd", "inference"): 0.0, ("regd", "prediction"): 1.0,
    ("regd_elem", "inference"): 1.0, ("regd_elem", "prediction"): 1.0,
}


def _task(benchmark):
    return "prediction" if benchmark.endswith("-prediction") else "inference"


def _regd_model_kwargs(method, task):
    kwargs = {"depth_on_containment": False, "margin": 0.0,
              "neg_sampling_gcis": ["gci2" if task == "prediction" else "gci0"]}
    if task == "prediction":
        kwargs["flip_gci2"] = True
    if method == "regd_elem":
        kwargs["radius"] = "softplus"
        kwargs["gci1_radius_term"] = False  # commented out in the code's Elem.nf2Loss
    kwargs["lamda"] = _LAMDA[(method, task)]
    return kwargs


def _build():
    out = []
    for onto, label in _ONTO.items():
        for task in ("inference", "prediction"):
            benchmark = f"regd-{onto}-{task}"
            common = {"task": task, "reference_ontology": label}
            for method in ("regd", "regd_elem"):
                out.append(Recipe(
                    benchmark=benchmark, method=method, source=YANG2025_REGD,
                    model_kwargs={**_regd_model_kwargs(method, task), "embed_dim": 10,
                                  "learning_rate": _LR[(method, onto, task)]},
                    training_regime="regd-code-training",
                    regime_kwargs={**common, "design": "regd",
                                   "host_loss": "mse" if method == "regd" else "mean"},
                    epochs=5000, validate_every=100, locator=_LOC,
                    notes=_NOTE + (" Balls: softplus radii in training, |radius| at test "
                                   "time, as in the code; p = 2 (the default)."
                                   if method == "regd_elem" else " Boxes: p = 1.")))
            out.append(Recipe(
                benchmark=benchmark, method="elbe", source=YANG2025_REGD,
                model_kwargs={"embed_dim": 10, "learning_rate": _LR[("elbe", onto, task)]},
                training_regime="regd-code-training",
                regime_kwargs={**common, "design": "baseline", "margin": 0.0,
                               "neg_target": 2.0},
                epochs=5000, validate_every=100, locator=_LOC, notes=_NOTE))
            out.append(Recipe(
                benchmark=benchmark, method="elem", source=YANG2025_REGD,
                model_kwargs={"embed_dim": 10, "learning_rate": _LR[("elem", onto, task)],
                              "gci1_radius_term": False},
                training_regime="regd-code-training",
                regime_kwargs={**common, "design": "baseline", "margin": 0.0},
                epochs=5000, validate_every=100, locator=_LOC,
                notes=_NOTE + " mOWL's ELEm keeps |radius|, a rectified negative loss and "
                              "its overlap loss for ∃r.C ⊑ D; the code's Elem differs in "
                              "all three (see the regime card)."))
    return tuple(out)


RECIPES = _build()
