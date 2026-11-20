"""The recipes of DELE: every (benchmark, method, DELE variant) of Tables 3-6 is
covered, resolves against the catalogue and builds its training regime."""

from unittest import TestCase

from mowl.benchmarks.entries.dele import BENCHMARKS
from mowl.benchmarks.recipes import list_recipes
from mowl.catalog import get_method, get_regime

#: The four benchmarks of DELE's Tables 3-6 (the human benchmarks have no published
#: numbers and no recipes).
_BENCH = {"dele-yeast-function", "dele-yeast-ppi", "dele-foodon-subsumption",
          "dele-galen-subsumption"}
_ALL = {b.key for b in BENCHMARKS}


class TestDELERecipes(TestCase):

    def test_every_table_cell_has_a_recipe(self):
        recipes = list_recipes(source="mashkova2024")
        found = {(r.benchmark, r.method) for r in recipes}
        expected = {(b, m) for b in _BENCH for m in ("elem", "elbe", "box2el")}
        self.assertTrue(_BENCH <= _ALL)
        self.assertEqual(found, expected)
        # three variants per cell (baseline, +l, +l+n) and the paper's epochs
        by_cell = {}
        for r in recipes:
            by_cell.setdefault((r.benchmark, r.method), []).append(r)
        for cell, rs in by_cell.items():
            self.assertEqual(len(rs), 3, cell)
            self.assertTrue(any("filter_negatives" in r.regime_kwargs for r in rs))
        for r in recipes:
            self.assertIn(r.benchmark, _BENCH)
            get_method(r.method)  # raises if unknown
            self.assertEqual(r.training_regime, "dele-code-training")
            card = get_regime(r.training_regime)
            self.assertEqual(card.kind.value, "training")
            expected_epochs = 18000 if r.benchmark.startswith("dele-yeast") else 800
            self.assertEqual(r.epochs, expected_epochs)
            self.assertEqual(r.validate_every, 9 if expected_epochs == 18000 else 1)

    def test_regime_builds(self):
        from mowl.benchmarks.runner import resolve_training_regime
        for r in list_recipes(source="mashkova2024"):
            regime = resolve_training_regime(r.regime_spec())
            keys = [p.key for p in regime.regimes]
            self.assertIn("dele-code-losses", keys)
            self.assertIn("per-normal-form-minibatch", keys)
            self.assertIn("reduce-lr-on-plateau", keys)
            if "filter_negatives" in r.regime_kwargs:
                self.assertIn("dele-closure-filtered-negatives", keys)
            else:
                self.assertNotIn("dele-closure-filtered-negatives", keys)
            if r.regime_kwargs.get("negatives") == "all":
                self.assertIn("dele-all-normal-form-negatives", keys)
