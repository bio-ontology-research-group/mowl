# Guide for AI coding agents

Read this before changing embedding methods, training, evaluation or benchmarks in mOWL.
`CLAUDE.md` has the general project rules (setup, tests, docs, release); this file
describes how the embedding code is organised and what a change must include.

## The four parts

A paper usually bundles a representation, a way to train it, a way to test it and some
datasets. mOWL keeps them apart. Every change must fit this split; do not add per-paper
special cases outside it.

| Part | What it is | Code | Catalogue card |
|---|---|---|---|
| Method | The representation: geometry and per-axiom losses | `mowl/nn/<logic>/<key>/`, `mowl/models/<key>/` | `mowl/catalog/entries/<key>.py` (`MethodCard`) |
| Training regime | How a model is trained, independent of the representation: negatives, batching, loss combination, margin meaning, schedule, model selection | `mowl/training/` (`TrainingRegime` subclasses) | `mowl/catalog/regimes/` (`RegimeCard`) |
| Evaluation regime | How a model is tested: test set, candidates, score, filtering, ties, metrics | `mowl/evaluation/` (`EvaluationRegime` subclasses) | `mowl/catalog/regimes/` (`RegimeCard`) |
| Benchmark | A dataset plus an evaluation regime, with the numbers printed in papers and the recipes behind them | `mowl/benchmarks/entries/`, `published/`, `recipe_sets/` | — |

A **recipe** (`mowl.benchmarks.Recipe`) is the model arguments, training regime and epochs
under which one publication obtained its numbers on one benchmark.

Cards, benchmarks, published results and recipes are discovered automatically from their
packages: adding one means adding a file that exposes `CARD`/`CARDS`, `REGIME`/`REGIMES`,
`RESULTS` or `RECIPES`.

## Papers are combinations, not limits

The point of the split is recombination: a model from one paper, the training regime of a
second and the evaluation of a third. A publication is recorded as one combination (a
recipe), which documents and reproduces it, but nothing in the library may assume that
combination. When you add a method, make every part of its training and testing that could
serve another method a shared regime, and state on its card:

- `expressivity`: the logic the implementation supports, in description logic notation
  (`mowl.logic.Expressivity`); a test compares it with what the module implements;
- `tasks`: the inference tasks it gives a score for (`mowl.logic.TASKS`).

Evaluation regime cards state the `task` they measure and their `target` (entailed test
axioms: deductive inference; held-out ones: inductive prediction). Logical ground truth
comes from a reasoner backend chosen by expressivity (`mowl.reasoning`). See
`docs/source/logic/index.rst`.

## Finding things

```bash
python -m mowl.benchmarks list                      # all benchmarks
python -m mowl.benchmarks show <benchmark>          # dataset, evaluation regime
python -m mowl.benchmarks published <benchmark>     # numbers printed in papers
python -m mowl.benchmarks run <benchmark> --method <key> --recipe --seed 0 --out r.json
python -m mowl.benchmarks compare r.json            # a result against the published ones
```

```python
from mowl.catalog import list_methods, get_method, list_regimes, regimes_of
from mowl.benchmarks import list_benchmarks, published_results, get_recipe, list_recipes
```

The catalogue does not need the JVM. Everything that imports model code does: call
`mowl.init_jvm("2g")` first.

Documentation to read for the area you touch:

- `docs/source/methods/adding_a_method.rst`: the checklist for a new method.
- `docs/source/methods/training_and_evaluation.rst`: how the literature trains and tests.
- `docs/source/training/index.rst`: training regimes, strict and slack margins.
- `docs/source/benchmarks/index.rst`: benchmarks, protocols, recipes.
- `docs/source/benchmarks/reproduction.rst`: which publications are reproduced, with numbers.

## Rules

1. **A difference from a paper's code is a named regime or a regime argument**, with a
   card and a test, and a recipe that uses it. It is not a flag hidden in a model.
2. **Known defects are fixed, and the fix is the default.** If a published number depends
   on a defect of the reference code, keep the old behaviour as an explicit option and
   let the recipe select it (example: `gci1_radius_term=False` for ELEmbeddings).
3. **Every guarantee on a card names its source** (publication and theorem or section)
   and the condition under which it holds. Do not record a claim you have not read.
4. **Check the reference code, not only the paper text**, before claiming a
   reproduction: margin sign, corrupted columns and number of negatives, fixed or redrawn
   negatives, what a step is, schedule, model selection, test-time score, candidates,
   filtering, ties.
5. **Never tune on the test set.** Report the number of seeds. Never claim a reproduction
   that was not run; update `docs/source/benchmarks/reproduction.rst` with the outcome.
6. **Optional dependencies** (a language model, a conic solver) are imported inside the
   functions that need them and are listed on the card, never in `install_requires`.
7. RST examples use `.. testcode::`, not `.. code-block:: python`. Update the
   `[Unreleased]` section of `CHANGELOG.md`. Do not set version strings.

## Tests

```bash
pytest -m "not slow"                 # what CI runs
pytest tests/catalog tests/benchmarks tests/training   # after touching cards, recipes, regimes
```

- A new loss needs a semantics test: zero when the geometric condition holds, positive
  when it does not, on hand-built configurations.
- A new regime, scorer or recipe needs a small test. `tests/benchmarks/test_registry.py`
  checks that all cards, published results and recipes are consistent.
- Tests that train for more than a few seconds are marked `@pytest.mark.slow`.

## Things that have gone wrong before

- The margin means different things in different code bases (slack: a violation up to the
  margin is free; strict: inclusion must hold with room to spare). Use
  `mowl.training.SlackMargin` / `StrictMargin`, never a raw sign change.
- The test-time score is often not the training loss. Scorers live in
  `mowl/evaluation/el_scoring.py`.
- The EL training loop is full-batch by default: one optimisation step per epoch.
  Reference codes that sample a batch per normal form need
  `mowl.training.PerNormalFormMiniBatches`.
- The normaliser caches its output next to the ontology file
  (`*_mowl_el_normalized_v2.owl`). Do not write to a dataset directory while another
  process is loading it.
