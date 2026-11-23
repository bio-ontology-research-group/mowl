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

    def test_get_recipe_selects_the_variant(self):
        from mowl.benchmarks.recipes import get_recipe
        with self.assertRaises(KeyError):
            get_recipe("dele-galen-subsumption", "elem", "mashkova2024")
        r = get_recipe("dele-galen-subsumption", "elem", "mashkova2024", "+l+n")
        self.assertTrue(r.regime_kwargs["filter_negatives"])
        self.assertEqual(r.model_kwargs["embed_dim"], 400)
        r = get_recipe("dele-galen-subsumption", "elem", "mashkova2024", "base")
        self.assertEqual(r.regime_kwargs["negatives"], "gci2")
        self.assertNotIn("filter_negatives", r.regime_kwargs)

    def test_runner_resolves_source_and_name(self):
        from mowl.benchmarks.runner import resolve_recipe
        r = resolve_recipe("dele-galen-subsumption", "elem", "mashkova2024:+l")
        self.assertEqual(r.name, "+l")
        r = resolve_recipe("box2el-go-prediction", "box2el", True)
        self.assertEqual(r.name, "")

    def test_regime_builds(self):
        from mowl.benchmarks.runner import resolve_training_regime
        for r in list_recipes(source="mashkova2024"):
            regime = resolve_training_regime(r.regime_spec())
            keys = [p.key for p in regime.regimes]
            self.assertIn("dele-code-losses", keys)
            self.assertIn("per-normal-form-minibatch", keys)
            self.assertIn("reduce-lr-on-plateau", keys)
            self.assertIn("dele-code-loss", keys)
            # fixed negatives, as the reference (np.random.seed(0) in every batch)
            negs = [p for p in regime.regimes if p.key == "dele-all-normal-form-negatives"]
            self.assertEqual(len(negs), 1)
            self.assertEqual(negs[0].fixed_seed, 0)
            if "filter_negatives" in r.regime_kwargs:
                self.assertIn("dele-closure-filtered-negatives", keys)
            else:
                self.assertNotIn("dele-closure-filtered-negatives", keys)
            if r.regime_kwargs.get("negatives") == "all":
                self.assertIn("dele-all-normal-form-negatives", keys)


def test_the_reference_reading_keeps_the_margin_of_the_box2el_negatives():
    """The losses of Box2EL are those of the DELE reference code already: with the reference
    reading switched on, the negative and disjointness losses of the combination of parts are
    those of the direct module (squared, which the reading leaves to the training loss).
    Switching their margin to a strict one, as for ELEmbeddings and ELBE, made the model stop
    after about twenty steps on FoodOn (Hits@10 0.0005 instead of 0.034)."""
    import types
    import torch as th
    from mowl.nn import BoxSquaredELModule
    from mowl.baselines.nn.el.boxsquaredel.module import BoxSquaredELModule as Direct
    from mowl.training.dele import DELECodeLosses
    th.manual_seed(0)
    direct = Direct(30, 3, embed_dim=16, gamma=0.1, delta=4, all_negative_losses=True)
    parts = BoxSquaredELModule(30, 3, embed_dim=16, gamma=0.1, delta=4,
                               all_negative_losses=True)
    parts.load_state_dict(direct.state_dict())
    DELECodeLosses().attach(types.SimpleNamespace(module=parts))
    pairs, triples = th.randint(0, 30, (40, 2)), th.randint(0, 30, (40, 3))
    triples[:, 1] %= 3
    for form, rows, neg, squared in (("gci0", pairs, True, False), ("gci1", triples, True, False),
                                     ("gci1_bot", triples, False, True)):
        ours = parts(rows, form, neg=True) if neg else parts(rows, form)
        theirs = direct(rows, form, neg=True) if neg else direct(rows, form)
        ours = ours.flatten() ** 2 if squared else ours.flatten()
        assert th.allclose(ours, theirs.flatten(), atol=1e-5), form
