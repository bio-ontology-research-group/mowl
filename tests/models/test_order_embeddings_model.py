"""Order embeddings (Vendrov et al., ICLR 2016) over the class taxonomy."""
import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

from mowl.evaluation.model_scoring import axiom_scoring_of  # noqa: E402
from mowl.models.order_embeddings import OrderEmbeddings  # noqa: E402
from mowl.nn.el.order_embeddings import OrderEmbeddingsModule  # noqa: E402
from tests.models.taxonomy_helpers import (check_trained_taxonomy, total_loss,  # noqa: E402
                                           train)


def test_violation_is_zero_iff_order_holds_and_order_is_transitive():
    E = OrderEmbeddingsModule.violation
    a = th.tensor([[3.0, 2.0]])      # a ⪯ b ⪯ c in the reversed product order
    b = th.tensor([[2.0, 2.0]])
    c = th.tensor([[1.0, 0.5]])
    assert E(a, b).item() == pytest.approx(0.0, abs=1e-9)
    assert E(b, c).item() == pytest.approx(0.0, abs=1e-9)
    assert E(a, c).item() == pytest.approx(0.0, abs=1e-9)           # transitivity
    assert E(b, a).item() == pytest.approx(1.0, abs=1e-6)           # ||max(0, a - b)||
    assert E(c, a).item() == pytest.approx((4 + 2.25) ** 0.5, abs=1e-6)
    # antisymmetry: both directions are zero only for equal vectors
    assert E(a, a).item() == pytest.approx(0.0, abs=1e-9)
    # the origin is the top element: everything is below it
    assert E(a, th.zeros(1, 2)).item() == pytest.approx(0.0, abs=1e-9)
    # incomparable vectors violate in both directions
    u, v = th.tensor([[2.0, 0.0]]), th.tensor([[0.0, 2.0]])
    assert E(u, v).item() > 1 and E(v, u).item() > 1


def test_embeddings_are_non_negative_and_losses_follow_eq_4():
    module = OrderEmbeddingsModule(2, embed_dim=2, margin=1.0)
    with th.no_grad():
        module.class_embed.weight.copy_(th.tensor([[-3.0, 2.0], [1.0, -0.5]]))
    assert (module.vectors(th.tensor([0, 1])) >= 0).all()      # |w|
    pair = th.tensor([[0, 1]])                                 # (3,2) ⪯ (1,.5): holds
    assert module(pair, "gci0").item() == pytest.approx(0.0, abs=1e-9)
    assert module(pair, "gci0", neg=True).item() == pytest.approx(1.0, abs=1e-6)  # margin
    rev = th.tensor([[1, 0]])
    e = module(rev, "gci0").item()
    assert e == pytest.approx((4 + 2.25) ** 0.5, abs=1e-6)
    assert module(rev, "gci0", neg=True).item() == 0.0         # E > margin


def test_trains_on_the_taxonomy_and_declares_scoring():
    model = train(OrderEmbeddings, epochs=300, embed_dim=8, learning_rate=0.05)
    assert axiom_scoring_of(model) == "module" and model.scored_forms == ("gci0",)
    check_trained_taxonomy(model)
    first = train(OrderEmbeddings, epochs=1, embed_dim=8, learning_rate=0.05)
    assert total_loss(model) < total_loss(first)
    # stated axioms are order pairs: zero violation after training
    assert max(float(model.subsumption_costs([s], [t])[0]) for s, t in
               [(model.class_index_dict["http://mowl.borg/bench#A"],
                 model.class_index_dict["http://mowl.borg/bench#B"])]) < 0.05


def test_ontology_without_subclass_axioms_is_refused():
    from mowl.datasets import PathDataset
    import os
    import tempfile
    d = tempfile.mkdtemp(prefix="order_empty_", dir=os.environ.get("TMPDIR"))
    path = os.path.join(d, "o.owl")
    with open(path, "w") as f:
        f.write("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\nOntology(<http://x/o>\n"
                "Declaration(Class(<http://x/A>))\nDeclaration(Class(<http://x/B>))\n)\n")
    model = OrderEmbeddings(PathDataset(path), embed_dim=2, model_filepath=d + "/m.pt")
    with pytest.raises(ValueError):
        model.train(1)
