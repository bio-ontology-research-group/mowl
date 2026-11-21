Evaluation with the deductive closure (DELE)
============================================

:class:`~mowl.evaluation.deductive.DELEEvaluation` is the evaluation regime of DELE
[mashkova2024dele]_ (Section 6.2, Tables 3-6). For axioms :math:`A \sqsubseteq B`
(``gci_name="gci0"``) or :math:`A \sqsubseteq \exists r.B` (``"gci2"``):

1. test axioms entailed by the training ontology are removed (only "novel" axioms are
   tested); alternatively ``test_set="closure"`` tests on the entailed but not asserted
   axioms;
2. each test axiom is ranked against all candidate classes :math:`B` using the model's
   loss as score (ties get the average rank);
3. **NF** metrics use the plain ranking, **F** metrics remove from the ranking every
   candidate entailed by the training and testing ontologies together (the test axiom
   itself stays);
4. Hits@n, macro and micro mean rank and AUC are reported for NF and F, plus the
   differences ``nf_minus_f_macro_mr`` and ``nf_minus_f_micro_mr``. Micro metrics are
   computed per subclass :math:`A` and averaged (DELE, Eqs. 19-22).

.. testcode::

   import torch as th
   from java.util import HashSet
   from mowl.datasets import Dataset
   from mowl.owlapi import OWLAPIAdapter
   from mowl.evaluation.deductive import DELEEvaluation

   ad = OWLAPIAdapter()
   ns = "http://example.org/"

   def ontology(iri, pairs):
       ont = ad.create_ontology(iri)
       axioms = HashSet()
       for a, b in pairs:
           axioms.add(ad.create_subclass_of(ad.create_class(ns + a), ad.create_class(ns + b)))
       ad.owl_manager.addAxioms(ont, axioms)
       return ont

   dataset = Dataset(ontology(ns + "train", [("A", "B"), ("B", "C")]),
                     testing=ontology(ns + "test", [("A", "D"), ("A", "C")]))
   classes = [ns + c for c in "ABCD"]

   class Scorer(th.nn.Module):            # any callable (data, gci_name) -> scores
       def forward(self, data, gci_name):
           return (data[:, 0] - data[:, 1]).abs().float()

   regime = DELEEvaluation(gci_name="gci0", heads=classes, tails=classes)
   metrics = regime.evaluate(Scorer(), dataset)["all"]
   print(metrics["num_test"], metrics["macro_mr"] >= metrics["f_macro_mr"])

.. testoutput::

   1 True

For a trained :class:`~mowl.base_models.EmbeddingELModel`, pass the model itself:
``regime.evaluate(model, dataset)`` scores with ``model.module``. The ranking itself is
available as :class:`~mowl.evaluation.deductive.DeductiveRankingEvaluator`, which accepts
explicit test and filter tensors and a :class:`~mowl.training.DeductiveClosure`.

Differences from the reference code: filtered candidates are removed from the ranking
instead of having their scores multiplied by :math:`10^6` (which leaves zero scores in
place and also moves the test axiom); AUC is computed in closed form so that average
ranks are allowed.
