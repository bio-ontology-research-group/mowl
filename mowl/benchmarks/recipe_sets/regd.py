"""Recipes of RegD (Yang and Chen, arXiv:2501.17518) and its ELBE / ELEm baselines, from
the paper and the authors' code (HuiYang1997/RegD, EXP_ontology)."""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries._refs import YANG2025_REGD

_LOC = ("EXP_ontology of HuiYang1997/RegD: baseline_sweep.py (5000 steps, batch 512, "
        "validation every 100) and run_all_sweeps.py (the sweep)")

_NOTE = (
    "The published numbers are the best of a learning-rate x margin sweep per dataset, "
    "task, model and dimension, judged by the test F1 that the code writes out; this "
    "recipe fixes one configuration (see the report of the reproduction task for the "
    "sweep). The recipe is written for embedding dimension 10; for dimension 5 run with "
    "--param embed_dim=5. The RegD models need the code's variants: the depth term only "
    "on non-contained pairs, negatives only on the task's normal form, and (prediction "
    "task) the reversed gci2 inclusion. The ball host's radii are absolute values in "
    "mOWL, softplus in the code; the code also clips the parent offsets at exp(-1e-4); "
    "neither is replicated.")


def _task(benchmark):
    return "prediction" if benchmark.endswith("-prediction") else "inference"


def _regd_kwargs(task):
    kwargs = {"depth_on_containment": False,
              "neg_sampling_gcis": ["gci2" if task == "prediction" else "gci0"]}
    if task == "prediction":
        kwargs["flip_gci2"] = True
    return kwargs


def _build():
    out = []
    for onto in ("galen", "go", "anatomy"):
        for task in ("inference", "prediction"):
            benchmark = f"regd-{onto}-{task}"
            for method, dim in (("regd", 10), ("regd_elem", 10)):
                out.append(Recipe(
                    benchmark=benchmark, method=method, source=YANG2025_REGD,
                    model_kwargs={**_regd_kwargs(task), "embed_dim": dim,
                                  "learning_rate": 0.005},
                    training_regime="regd-code-training",
                    regime_kwargs={"task": task, "design": "regd"},
                    epochs=5000, validate_every=100, locator=_LOC,
                    notes=_NOTE + " For regd_elem the paper finds p = 2 (the default)."
                ))
            out.append(Recipe(
                benchmark=benchmark, method="elbe", source=YANG2025_REGD,
                model_kwargs={"embed_dim": 10, "learning_rate": 0.005},
                training_regime="regd-code-training",
                regime_kwargs={"task": task, "design": "baseline", "margin": 0.05,
                               "neg_target": 2.0},
                epochs=5000, validate_every=100, locator=_LOC, notes=_NOTE))
            out.append(Recipe(
                benchmark=benchmark, method="elem", source=YANG2025_REGD,
                model_kwargs={"embed_dim": 10, "learning_rate": 0.0005},
                training_regime="regd-code-training",
                regime_kwargs={"task": task, "design": "baseline", "margin": 0.05},
                epochs=5000, validate_every=100, locator=_LOC, notes=_NOTE))
    return tuple(out)


RECIPES = _build()
