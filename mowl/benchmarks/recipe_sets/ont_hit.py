"""Recipes of OnT and of its HiT baseline (Yang et al., ISWC 2025), from the paper
(Section 5.1) and the reference code (HuiYang1997/OnT, commit 9a2cdc3)."""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries.ont import YANG2025ONT

#: The centripetal weight grid searched by the reference code (ont/evaluation/,
#: ``range(20) / 10``); the paper's Section 5.1 states {0, 0.1, ..., 1}.
_CODE_GRID = [i / 10 for i in range(20)]

_LOC = ("config.yaml and ont/data/load.py of HuiYang1997/OnT; Section 5.1 "
        "'Experimental Protocol' of the paper")

_NOTE = (
    "The settings are those of Section 5.1 (all-MiniLM-L12-v2, 1 epoch, 1 negative per "
    "axiom, α = 3.0, β = 0.5, learning rate 1e-5), with the batch size 64 and the "
    "warm-up 500 steps of config.yaml. The λ grid follows the code ({0.0, ..., 1.9}), "
    "not the paper ({0, ..., 1}). verbalisation_style='release' reproduces the texts of "
    "the data release (see the ont-* benchmark notes). Deviations, same distribution: "
    "the negatives are sampled by mOWL when training starts instead of read from the "
    "release (whose loader keeps only the first of the 10 stored negatives anyway), and "
    "the release's training/validation splits are smaller than the paper's Table 1. The "
    "numbers were produced with the authors' normalised data; results are not "
    "comparable with the box2el-* benchmarks.")

_HIT_NOTE = (
    "HiT was run by the OnT authors with OnT's settings (Section 5.1), restricted to "
    "the atomic (NF1) axioms ('a language model-based method limited to taxonomic "
    "structures'). mOWL's HiT therefore trains with OnT's margins α = 3.0, β = 0.5 and "
    "batch size 64, not with its own paper's defaults (α = 5.0, β = 0.1, batch 256, 10 "
    "negatives, 20 epochs). Remaining difference: mOWL's HiT excludes known ancestors "
    "from the negatives (HiT's closed-world rule); the OnT pipeline's negatives are "
    "unfiltered uniform draws. " + _NOTE)


def _build():
    out = []
    for name in ("galen", "go", "anatomy"):
        for task in ("prediction", "inference"):
            kwargs = {"encoder": "sentence-transformers/all-MiniLM-L12-v2",
                      "verbalisation_style": "release", "learning_rate": 1e-5,
                      "batch_size": 64, "num_negatives": 1, "warmup_steps": 500,
                      "centri_weight_grid": list(_CODE_GRID)}
            out.append(Recipe(
                benchmark=f"ont-{name}-{task}", method="ont", source=YANG2025ONT,
                model_kwargs={**kwargs, "clustering_margin": 3.0,
                              "centripetal_margin": 0.5},
                training_regime="ont-code-training",
                epochs=1, locator=_LOC, notes=_NOTE))
            out.append(Recipe(
                benchmark=f"ont-{name}-{task}", method="hit", source=YANG2025ONT,
                model_kwargs={**kwargs, "clustering_margin": 3.0,
                              "centripetal_margin": 0.5},
                training_regime="ont-code-training",
                epochs=1, locator=_LOC + "; HiT baseline of yang2025ont",
                notes=_HIT_NOTE))
    return tuple(out)


RECIPES = _build()
