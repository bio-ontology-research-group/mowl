Logical faithfulness
====================

Ranking metrics such as MRR or Hits@k treat every axiom that is not in the test set as
false. For an ontology this is wrong in two ways: a candidate axiom can be **entailed**,
**contradictory** or **undetermined** (open world assumption), and a model that ranks a
logically impossible axiom as high as a plausible new one is not merely inaccurate but
semantically wrong. Mashkova & Hoehndorf [mashkova2026]_ define four nested criteria for
a score function :math:`f` over candidate axioms:

* **L0 discrimination**: every entailed axiom outscores every contradictory one;
* **L1 logical admissibility**: no contradictory axiom outscores any consistent one;
* **L2 monotonic logical faithfulness (MLF)**:
  :math:`f(\alpha^-) < f(\alpha^?) < f(\alpha^+)` for contradictory, undetermined and
  entailed axioms;
* **L3 probabilistic logical faithfulness (PLF)**: :math:`f` respects the relative model
  count :math:`P(\alpha\mid\mathcal{O}) = \#(\mathcal{O}\cup\{\alpha\})/\#(\mathcal{O})`.

PLF implies MLF, which implies admissibility, which implies discrimination (Theorem 1 of
the paper). This matters for region embeddings: whether a model handles conjunction and
disjointness correctly shows up per normal form (the paper finds disjointness handled
well and conjunctive subsumption inverted for ball embeddings), not in an average rank.

The module :mod:`mowl.evaluation.faithfulness` implements

* :class:`~mowl.evaluation.faithfulness.LogicalStatusOracle`: the exact status of a
  candidate axiom of each normal form (``gci0`` :math:`C \sqsubseteq D`, ``gci1``
  :math:`C \sqcap D \sqsubseteq E`, ``gci2`` :math:`C \sqsubseteq \exists r.D`, ``gci3``
  :math:`\exists r.C \sqsubseteq D`, ``gci1_bot`` :math:`C \sqcap D \sqsubseteq \bot`),
  decided by :math:`\mathcal{EL}^{++}` saturation;
* :func:`~mowl.evaluation.faithfulness.generate_test_sets`: test sets :math:`T_+, T_-,
  T_?` sampled from the candidate space;
* all metrics of the paper: :math:`AUC_{+/-}` (L0), Adm@k (L1), FVR and Pair-FVR (L2),
  :math:`\rho_P` (L3), MRR and Hits@k, and the four criteria as booleans on the test
  sets;
* :class:`~mowl.evaluation.faithfulness.RelativeModelCounter`: exact relative model
  counts over a bounded Herbrand domain (built-in counter for small ontologies; the
  optional ``pyganak`` package for larger ones);
* the evaluation regimes :class:`~mowl.evaluation.faithfulness.FaithfulnessHierarchy` and
  :class:`~mowl.evaluation.faithfulness.LossModelness`.

What "contradictory" means
--------------------------

Definition 1 of the paper calls :math:`\alpha` contradictory if
:math:`\mathcal{O}\cup\{\alpha\}` is inconsistent. A TBox without individuals is
consistent with every :math:`\mathcal{EL}` axiom (all concepts may be empty), so the
paper's test sets are built under a non-emptiness assumption. mOWL makes it explicit:
with ``witnesses="paper"`` (default) every concept name, every conjunction on the left
of an axiom and every existential occurring in the ontology is assumed non-empty unless
the ontology proves it empty, and :math:`\alpha` is contradictory if it would make one of
them empty. ``witnesses="names"`` assumes only the concept names non-empty, and
``witnesses="none"`` is Definition 1 read literally (classical consistency, which uses
the ABox).

Example
-------

The working example of the paper (Appendix D) with a hand-written score function:

.. testcode::

   import mowl
   mowl.init_jvm("2g")
   from mowl.geometric.normal_forms import NF1, NF2, BOTTOM
   from mowl.evaluation.faithfulness import (LogicalStatusOracle, generate_test_sets,
                                             evaluate_faithfulness, RelativeModelCounter)

   ns = "http://example.org/"
   phd, student, prof, person, ta = (ns + n for n in
                                     ("PhD", "Student", "Professor", "Person", "TA"))
   ontology = [NF1(phd, student), NF2(student, prof, BOTTOM), NF1(student, person),
               NF1(prof, person), NF1(ta, person)]

   oracle = LogicalStatusOracle(ontology)
   print(oracle.status("gci0", (phd, student)))
   print(oracle.status("gci0", (phd, prof)))
   print(oracle.status("gci0", (ta, student)))

   test_sets = generate_test_sets(oracle, normal_forms=["gci0"], n_entailed=10,
                                  n_contradictory=10, n_undetermined=10)
   counter = RelativeModelCounter(ontology, domain_size=2, backend="python")
   ts = test_sets["gci0"]
   ts.p_values = {a: float(counter.relative_count("gci0", a)) for a in ts.undetermined}

   # A naive score: 1 if the names share a prefix of three characters, else 0.
   def score(axioms):
       return [float(c[len(ns):][:3] == d[len(ns):][:3]) for c, d in axioms]

   result = evaluate_faithfulness(score, test_sets, oracle.classes, k=2)["gci0"]
   print(sorted(k for k in result if not k.startswith("n_")))

.. testoutput::

   entailed
   contradictory
   undetermined
   ['adm@2', 'admissible', 'auc_pm', 'discriminative', 'fvr', 'hits@1', 'hits@10', 'hits@100', 'monotonic_faithful', 'mrr', 'pair_fvr', 'probabilistic_faithful', 'rho_p']

Any trained :class:`~mowl.base_models.EmbeddingELModel` can be passed instead of
``score``: it is scored with the loss of each normal form (lower loss, higher
plausibility), as in the paper. Graph-based or language-model methods can be evaluated
through a function ``score(axioms)`` or ``score(normal_form, axioms)``.

On a :class:`~mowl.datasets.Dataset`, ``FaithfulnessHierarchy(n_per_class=1000).evaluate(model, dataset)`` does everything at
once and returns ``{normal_form: {metric: value}}``.

Benchmarks
----------

The benchmarks ``faithfulness-pizza`` and ``faithfulness-goplus`` use the released data
of the paper (:class:`~mowl.datasets.builtin.faithfulness.PizzaFaithfulnessDataset`,
:class:`~mowl.datasets.builtin.faithfulness.GOPlusFaithfulnessDataset`), one training
split per normal form, with the published relative model counts, and record the numbers
of the paper's Tables 4--13 for ELEmbeddings, EmEL++, Box2EL, TransE, DistMult, ComplEx
and BoxE.

Model-ness and the properties of Bourgaux et al.
------------------------------------------------

:func:`~mowl.evaluation.faithfulness.loss_modelness` reports, per normal form, the
fraction of training axioms whose loss with zero margin vanishes, i.e. whether the
embedding is a model of the training axioms as far as its loss can tell.
:mod:`mowl.evaluation.properties` checks, for an embedding with a satisfaction oracle
(for example :class:`mowl.geometric.ModelChecker`), the properties of Bourgaux et al.
(KR 2024) on a finite candidate set: being a model, entailment closure (Definition 4),
weak TBox faithfulness (only axioms consistent with the ontology hold, Definition 5) and
strong TBox faithfulness (only entailed axioms hold, Definition 6). Thresholding a score
connects the two views: L1 means some threshold yields an entailment-closed, weakly
faithful acceptance set on the test axioms; L2 means some threshold yields an
entailment-closed, strongly faithful one.
