"""Protocol binding :class:`mowl.evaluation.faithfulness.FaithfulnessHierarchy` to the
faithfulness benchmarks."""

from mowl.benchmarks.protocols.base import Protocol


class FaithfulnessProtocol(Protocol):
    """Evaluates a model with the faithfulness hierarchy of Mashkova & Hoehndorf (2026).

    The benchmark datasets have one training split per normal form, so one run trains on
    the split selected by the dataset keyword ``normal_form`` (default ``"gci0"``) and
    reports that subset only: :attr:`subsets` is narrowed to it. To evaluate another
    normal form, load the benchmark with ``dataset_kwargs={"normal_form": ...}``. On a
    dataset without released test sets (e.g. a small ontology of one's own) the test
    sets are generated with the reasoner and every normal form is reported.

    Keyword arguments are passed to
    :class:`~mowl.evaluation.faithfulness.FaithfulnessHierarchy`.
    """

    key = "faithfulness-hierarchy"

    def __init__(self, benchmark, **kwargs):
        super().__init__(benchmark, **kwargs)
        nf = benchmark.dataset_kwargs.get("normal_form")
        if nf is not None:
            self.subsets = (nf,)
        self.info = {}

    def _regime(self, dataset):
        from mowl.evaluation.faithfulness import NORMAL_FORMS, FaithfulnessHierarchy
        kwargs = dict(self.kwargs)
        nf = getattr(dataset, "normal_form", None)
        if nf is not None and "normal_forms" not in kwargs:
            kwargs["normal_forms"] = (nf,)
        regime = FaithfulnessHierarchy(**kwargs)
        self.subsets = tuple(s for s in regime.normal_forms if s in NORMAL_FORMS)
        return regime

    def evaluate(self, model, dataset):
        """:return: ``{normal_form: {metric: value}}`` restricted to the benchmark's
            metrics; metrics that are undefined on a split (no contradictory set) are
            ``None``."""
        results = self._regime(dataset).evaluate(model, dataset)
        out = {}
        for nf, res in results.items():
            self.info[nf] = {k: res[k] for k in ("n_entailed", "n_contradictory",
                                                 "n_undetermined") if k in res}
            out[nf] = {m: res.get(m) for m in self.metrics}
        return out
