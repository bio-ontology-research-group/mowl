"""A model saved by the direct implementation (ELBE, TransBox, BoxEL, BoxSEL, ELK-Em) loads
into the class users get, and the other way round the class users get saves what it loads."""
import os

import pytest
import torch as th

from tests.baselines.harness import default_dataset
from tests.baselines.pairs_boxrel import PAIRS

KEYS = ("elbe", "elbe/inclusion-joint", "transbox", "transbox/reference", "boxel",
        "boxel/signed-scale", "boxsel", "elkem", "elkem/reference")


@pytest.mark.parametrize("pair", [p for p in PAIRS if p.key in KEYS], ids=lambda p: p.key)
def test_saved_models_of_the_direct_implementation_load(pair, tmp_path):
    dataset = default_dataset()
    th.manual_seed(3)
    old = pair.old(dataset)
    th.manual_seed(4)
    new = pair.new(dataset)
    path = str(os.path.join(tmp_path, "model.pt"))
    th.save(old.module.state_dict(), path)
    new._model_filepath = path
    new.load_best_model()
    assert not new.module.training
    for name in pair.parameters(old):
        assert th.equal(pair.parameters(old)[name], pair.parameters(new)[name]), name
    # and the combination's own checkpoint loads into a fresh combination
    th.save(new.module.state_dict(), path)
    again = pair.new(dataset)
    again._model_filepath = path
    again.load_best_model()
    for name in pair.parameters(new):
        assert th.equal(pair.parameters(new)[name], pair.parameters(again)[name]), name
