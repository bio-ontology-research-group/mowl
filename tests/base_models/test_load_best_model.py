"""``load_best_model`` reloads a module a training regime has given a parameter of its own."""
import os
import tempfile

import pytest
import torch as th

from mowl.datasets.builtin import FamilyDataset


@pytest.mark.parametrize("path", ["mowl.models.ELEmbeddings",
                                  "mowl.baselines.models.elembeddings.model.ELEmbeddings"])
def test_a_parameter_added_by_a_regime_survives_the_reload(path):
    import importlib
    module_path, name = path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), name)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as folder:
        model = cls(FamilyDataset(), embed_dim=4,
                    model_filepath=os.path.join(folder, "model.pt"))
        model.module.register_parameter("subclass_relation", th.nn.Parameter(th.ones(4)))
        th.save(model.module.state_dict(), model.model_filepath)
        with th.no_grad():
            model.module.subclass_relation.zero_()
        model.load_best_model()
        assert th.equal(model.module.subclass_relation, th.ones(4))
        # a module without the parameters of the file is built anew, as before
        th.save({k: v for k, v in model.module.state_dict().items()
                 if k != "subclass_relation"}, model.model_filepath)
        model.load_best_model()
        assert not hasattr(model.module, "subclass_relation")
