"""Speed and numbers of the shared base of the hierarchy-embedding methods
(:class:`mowl.models.order_base.OrderEmbeddingModel`): negative sampling is vectorised, so an
epoch is linear in the number of edges, and gives the same negatives and parameters as the
earlier row-by-row loop."""
import os
import random
import tempfile
import time

import numpy as np
import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset  # noqa: E402
from mowl.models import DiskEmbeddings, HyperbolicCones, ShadowCones  # noqa: E402


def random_tree_dataset(n, extra_edges=0, seed=0):
    rnd = random.Random(seed)
    directory = tempfile.mkdtemp(prefix="mowl_speed_", dir=os.environ.get("TMPDIR"))
    path = os.path.join(directory, "tree.owl")
    with open(path, "w") as f:
        f.write("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\nOntology(<http://x/t>\n")
        for i in range(1, n):
            f.write(f"SubClassOf(<http://x/c{i}> <http://x/c{rnd.randrange(0, i)}>)\n")
        f.write(")\n")
    return PathDataset(path)


def old_generate_negatives(model, gci_name, gci_dataset, max_resample=10):
    """The row-by-row implementation that was in the model before it was vectorised."""
    from mowl.base_models.elmodel import EmbeddingELModel
    neg = EmbeddingELModel.generate_negatives(model, gci_name, gci_dataset)
    n = len(gci_dataset)
    n_named = len(model.named_class_index_dict)
    ids = np.array(list(model.named_class_index_dict.values()))
    columns = model.get_negative_sampling_config()[gci_name]["corrupt_column"]
    for _ in range(max_resample):
        codes = (neg[:, 0] * n_named + neg[:, 1]).tolist()
        bad = [i for i, c in enumerate(codes)
               if c in model._known_pairs or neg[i, 0] == neg[i, 1]]
        if not bad:
            break
        for i in bad:
            neg[i, columns[i // n]] = int(np.random.choice(ids))
    return neg


def _model(ds, cls=HyperbolicCones, **kw):
    return cls(ds, embed_dim=10, seed=0, **kw)


@pytest.mark.parametrize("cls", [HyperbolicCones, DiskEmbeddings, ShadowCones])
def test_vectorised_negatives_equal_the_row_loop(cls):
    ds = random_tree_dataset(60)
    model = _model(ds, cls, num_neg=10)
    data = model.training_datasets["gci0"]
    # a tiny class pool makes collisions with told pairs and self-pairs frequent
    for seed in range(3):
        np.random.seed(seed)
        new = model.generate_negatives("gci0", data)
        np.random.seed(seed)
        old = old_generate_negatives(model, "gci0", data)
        assert th.equal(new, old)
        codes = (new[:, 0] * len(model.named_class_index_dict) + new[:, 1]).tolist()
        assert not any(c in model._known_pairs for c in codes)


def test_training_steps_equal_those_of_the_row_loop(monkeypatch):
    ds = random_tree_dataset(40)

    def run(patch):
        m = _model(ds, HyperbolicCones, num_neg=3)
        if patch:
            monkeypatch.setattr(
                type(m), "generate_negatives",
                lambda self, g, d, max_resample=10: old_generate_negatives(self, g, d))
        np.random.seed(1)
        th.manual_seed(1)
        m.train(3)
        monkeypatch.undo()
        return m.module.class_embed.weight.detach().clone()

    assert th.allclose(run(False), run(True), atol=1e-6)


@pytest.mark.slow
def test_epoch_on_20000_classes_is_fast():
    ds = random_tree_dataset(20000)
    model = _model(ds, HyperbolicCones, num_neg=10, batch_size=10 ** 6)
    model.training_datasets
    model.train(1)  # warm up
    start = time.time()
    model.train(2)
    per_epoch = (time.time() - start) / 2
    # about 3 s on one CPU thread; the row loop needed more than 20 s here on a GPU-less box
    assert per_epoch < 30, per_epoch
