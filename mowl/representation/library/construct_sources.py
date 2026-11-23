"""Sources of constructed methods (:class:`~mowl.representation.construction.Source`).

==================  ===============  ==========================================================
key                 kind             builds
==================  ===============  ==========================================================
canonical model     completion       the canonical model of a normalised EL++ ontology
least model         least model      the least model of indexed EL++ tables
finite model        given            a given finite interpretation, as it is
boolean atoms       closed form      the atoms of a Boolean ALC TBox (JAIR Props. 7, 8)
lasso model         search           an ultimately periodic model of a Boolean LTL_ALC
                                     ontology (bounded search)
attributed          grounding        the canonical model of the grounding of an attributed
                                     ontology (Bourgaux, Ozaki & Pan, DL 2021)
box milp            solver           a box model found by a mixed-integer programme
                                     (Leemhuis & Kutz, NeSy 2025)
==================  ===============  ==========================================================

The sources build; they read nothing as regions. The mathematics they run (completion,
least-model saturation, model search, grounding, the MILP) lives in :mod:`mowl.geometric`,
which is the neutral side of these methods.
"""

import itertools

from mowl.representation.construction import (FiniteModel, Guarantee, Source,
                                              TemporalModel, register_source)

__all__ = ["CanonicalModelSource", "LeastModelSource", "FiniteModelSource",
           "BooleanAtomsSource", "LassoSource", "AttributedSource", "BoxMILPSource",
           "canonical_model_of"]


def _normalised_ontology_class():
    from mowl.geometric.normal_forms import NormalisedOntology
    return NormalisedOntology


def canonical_model_of(data):
    """The :class:`~mowl.geometric.completion.CanonicalModel` of a dataset, an OWLAPI
    ontology, a normalised ontology, or itself."""
    from mowl.geometric.completion import CanonicalModel
    if isinstance(data, CanonicalModel):
        return data
    if isinstance(data, _normalised_ontology_class()):
        return CanonicalModel.from_normalised(data)
    if hasattr(data, "ontology") and hasattr(data, "classes"):          # a mOWL dataset
        return CanonicalModel.from_dataset(data)
    return CanonicalModel.from_ontology(data)


def reads_ontology(data):
    """Whether :func:`canonical_model_of` can read ``data``: a mOWL dataset, an OWL API
    ontology, a normalised ontology or a canonical model."""
    from mowl.geometric.completion import CanonicalModel
    return (isinstance(data, CanonicalModel) or isinstance(data, _normalised_ontology_class())
            or (hasattr(data, "ontology") and hasattr(data, "classes"))
            or hasattr(data, "getAxioms"))


def _rejects_non_ontology(source, data):
    if reads_ontology(data):
        return None
    return (f"it reads an ontology (a mOWL dataset, an OWL API ontology, a normalised "
            f"ontology or a canonical model), not a {type(data).__name__}")


_EL = "normalised EL++ (NF1-NF4, NF6, NF7, bottom forms) with atomic ABox assertions"


@register_source
class CanonicalModelSource(Source):
    """The canonical model of a normalised EL++ ontology, by the ELK-style completion of
    :mod:`mowl.geometric.completion`: one element per satisfiable concept or nominal name.

    :param nominals: Whether the nominal names ``N_a`` of the individuals are concepts of
        the finite model (their extensions; the individuals are always there).
    """

    key = "canonical model"
    kind = "completion"
    logic = _EL
    reasoner = "EL++ completion (mowl.geometric.completion.Completion)"
    guarantees = (
        Guarantee("soundness", "The canonical model is a model of the normalised ontology.",
                  _EL),
        Guarantee("strong_faithfulness",
                  "A ⊑ B holds iff it is entailed; A is empty iff it is unsatisfiable "
                  "(concept names).", _EL),
        Guarantee("strong_faithfulness",
                  "Instance checking is sound and complete through the nominal elements.",
                  "ELH⊥ without nominals, functionality and range axioms"),
    )

    def __init__(self, nominals=True):
        self.nominals = bool(nominals)

    def rejects(self, data):
        return _rejects_non_ontology(self, data)

    def build(self, data):
        return FiniteModel.from_canonical_model(canonical_model_of(data), self.nominals)


@register_source
class LeastModelSource(Source):
    """The least model of indexed EL++ tables (:func:`mowl.geometric.least_model.least_model`):
    one element per class and per individual, merged into blocks if asked. The domain of
    the finite model is the set of live blocks (a dead block derives ⊥ and is no element);
    classes, roles and individuals are named by their ids.

    ``build`` takes a :class:`~mowl.geometric.least_model.LeastModel`, a dictionary of the
    arguments of ``least_model`` (``nb_classes``, ``gci0``, ...), or an indexed EL model
    (``least_model_from_model``).

    :param options: Further arguments of ``least_model`` (``method``, ``element_map``, ...).
    """

    key = "least model"
    kind = "least model"
    logic = "EL++ normal forms over indexed tables (with top/bottom ids, inverses)"
    reasoner = "sparse least-model saturation (mowl.geometric.least_model)"
    guarantees = (
        Guarantee("soundness", "The pruned least model is a model of the tables.",
                  "EL++ tables"),
        Guarantee("strong_faithfulness",
                  "A ⊑ B holds iff it follows from the tables by the completion rules "
                  "(the least model is the least fixed point).", "EL++ tables",
                  "with the identity element map"),
    )

    def __init__(self, **options):
        self.options = options

    def rejects(self, data):
        from mowl.geometric.least_model import LeastModel
        if isinstance(data, (LeastModel, dict)) or (
                hasattr(data, "nb_classes") and hasattr(data, "nb_rels")):
            return None
        return (f"it reads a LeastModel, a dictionary of indexed tables or an indexed model "
                f"(nb_classes, nb_rels, ...), not a {type(data).__name__}")

    def build(self, data):
        from mowl.geometric.least_model import LeastModel, least_model, least_model_from_model
        if isinstance(data, LeastModel):
            lm = data
        elif isinstance(data, dict):
            lm = least_model(**{**data, **self.options})
        else:
            lm = least_model_from_model(data, **self.options)
        return FiniteModel.of(lm)

    def default_names(self, built):
        lm = built.origin
        return (list(range(lm.nb_classes)), list(range(lm.nb_rels)),
                list(range(lm.nb_individuals)))


@register_source
class FiniteModelSource(Source):
    """A given finite interpretation, as it is (any of :meth:`FiniteModel.of
    <mowl.representation.construction.FiniteModel.of>`). The construction of
    Gutiérrez-Basulto & Schockaert (KR 2018, Prop. 3) and the al-cone construction from a
    finite model start here: they need a finite model, not the ontology."""

    key = "finite model"
    kind = "given"
    logic = "whatever the given finite model is a model of"
    reasoner = "none (the model is given)"
    guarantees = ()

    def build(self, data):
        from mowl.geometric.completion import CanonicalModel
        if isinstance(data, CanonicalModel) or (hasattr(data, "ontology")
                                                and hasattr(data, "classes")):
            # an ontology: the finite interpretation of its canonical model that the
            # convex-regions construction takes (concept names without nominals, no
            # individuals)
            from mowl.geometric.convex_regions import FiniteInterpretation
            data = FiniteInterpretation.from_canonical_model(canonical_model_of(data))
        return FiniteModel.of(data)


@register_source
class BooleanAtomsSource(Source):
    """The algebraic atoms of the Lindenbaum-Tarski algebra of a Boolean ALC TBox (the
    complete conjunctions of literals that satisfy every GCI), enumerated exhaustively;
    each atom is an element (Özçep, Leemhuis & Wolter, JAIR 2023, Prop. 7).

    Individuals: with ``faithful=False`` the first atom consistent with its assertions
    (Prop. 7, Example 1); with ``faithful=True`` the set of *all* atoms consistent with
    them (Prop. 8: strongly concept-faithful, read in a doubled space).

    :param names: Extra concept names of the signature.
    :param max_names: Refuse larger signatures (the atoms are enumerated).
    """

    key = "boolean atoms"
    kind = "closed form"
    logic = "Boolean ALC (no roles): GCIs and concept assertions"
    reasoner = "exhaustive enumeration of the atoms of the TBox"
    guarantees = (
        Guarantee("strong_faithfulness",
                  "C ⊑ D holds iff the TBox entails it (TBox-faithful).", "Boolean ALC"),
        Guarantee("strong_faithfulness",
                  "With faithful=True: a ∈ C holds iff the ontology entails C(a), for "
                  "every Boolean concept C.", "Boolean ALC", "faithful=True"),
    )

    def __init__(self, faithful=False, names=(), max_names=16):
        self.faithful = bool(faithful)
        self.names = tuple(names)
        self.max_names = int(max_names)

    def rejects(self, data):
        import collections.abc
        if isinstance(data, collections.abc.Iterable):
            return None
        return (f"it reads a list of the axiom objects of mowl.geometric.cones (GCI, "
                f"ConceptAssertion), not a {type(data).__name__}")

    def build(self, data):
        from mowl.geometric.cones import (GCI, ConceptAssertion, RoleAssertion, has_roles,
                                          signature, _eval_boolean)
        from mowl.geometric.errors import InconsistentOntology, UnsupportedAxiom
        axioms = list(data)
        tbox, abox = [], {}
        for ax in axioms:
            if isinstance(ax, RoleAssertion) or \
                    (isinstance(ax, GCI) and (has_roles(ax.sub) or has_roles(ax.sup))) or \
                    (isinstance(ax, ConceptAssertion) and has_roles(ax.concept)):
                raise UnsupportedAxiom(f"Not a Boolean ALC axiom: {ax}")
            if isinstance(ax, GCI):
                tbox.append(ax)
            else:
                abox.setdefault(ax.individual, []).append(ax.concept)
        sig, _, _ = signature(axioms)
        sig = sorted(sig | set(self.names))
        if len(sig) > self.max_names:
            raise ValueError(f"{len(sig)} concept names exceed max_names={self.max_names}.")
        types = []
        for bits in itertools.product((False, True), repeat=len(sig)):
            val = dict(zip(sig, bits))
            if all((not _eval_boolean(g.sub, val)) or _eval_boolean(g.sup, val) for g in tbox):
                types.append(val)
        if not types:
            raise InconsistentOntology("The Boolean TBox is unsatisfiable (no atom).")
        individuals = {}
        for ind, cs in abox.items():
            ok = [t for t in range(len(types)) if all(_eval_boolean(c, types[t]) for c in cs)]
            if not ok:
                raise InconsistentOntology(f"Assertions on {ind!r} are unsatisfiable.")
            individuals[ind] = frozenset(ok) if self.faithful else ok[0]
        concepts = {a: {t for t, val in enumerate(types) if val[a]} for a in sig}
        return FiniteModel(len(types), concepts, {}, individuals, origin=axioms)


@register_source
class LassoSource(Source):
    """An ultimately periodic classical model of a role-free temporal ontology
    (:func:`mowl.geometric.temporal_cones.find_lasso_model`, a bounded search), or a given
    :class:`~mowl.geometric.temporal_cones.LassoModel`: one finite model per time point
    over the domain of traces (Leemhuis, FCR 2024, Theorem 1).

    :param max_prefix, max_period, max_bits, witnesses: The bounds of the search.
    """

    key = "lasso model"
    kind = "search"
    logic = "Boolean LTL_ALC (role-free; next, until, eventually, always)"
    reasoner = "bounded search of ultimately periodic models (not a decision procedure)"
    produces = TemporalModel
    guarantees = (
        Guarantee("soundness", "The model found satisfies every axiom of the ontology.",
                  "Boolean LTL_ALC", "if a model is found within the bounds"),
    )

    def __init__(self, max_prefix=2, max_period=3, max_bits=14, witnesses=True):
        self.max_prefix, self.max_period = int(max_prefix), int(max_period)
        self.max_bits, self.witnesses = int(max_bits), bool(witnesses)

    def rejects(self, data):
        from mowl.geometric.temporal_cones import LassoModel, TemporalOntology
        if isinstance(data, (LassoModel, TemporalOntology)):
            return None
        return (f"it reads a TemporalOntology (or a LassoModel) of mowl.geometric."
                f"temporal_cones, not a {type(data).__name__}")

    def build(self, data):
        from mowl.geometric.temporal_cones import LassoModel, find_lasso_model
        model = data
        if not isinstance(data, LassoModel):
            model = find_lasso_model(data, self.max_prefix, self.max_period, self.max_bits,
                                     self.witnesses)
            if model is None:
                raise ValueError("No ultimately periodic model found within the given "
                                 "bounds.")
        names = sorted(set().union(*(s for tr in model.traces for s in tr)))
        steps = []
        for t in range(model.length):
            concepts = {n: {d for d, tr in enumerate(model.traces) if n in tr[t]}
                        for n in names}
            steps.append(FiniteModel(len(model.traces), concepts, {},
                                     dict(model.individuals)))
        return TemporalModel(steps, model.loop_start, dict(model.individuals), origin=model)

    def default_names(self, built):
        return sorted(built.steps[0].concepts), [], sorted(built.individuals)


@register_source
class AttributedSource(Source):
    """The canonical model of the grounding ``dl(O_g)`` of a ground (temporally) attributed
    ontology (:func:`mowl.geometric.attributed.ground`; Bourgaux, Ozaki & Pan, DL 2021,
    proof of Theorem 6). A name of the geometry is a pair ``(concept, specifier)``, read
    as the grounded name ``concept@specifier``."""

    key = "attributed"
    kind = "grounding"
    logic = ("ground DL-Lite^{H,@}_horn without negative role inclusions; temporal "
             "attributes time, during, since, until")
    reasoner = "grounding (mowl.geometric.attributed.ground), then EL++ completion"
    guarantees = (
        Guarantee("soundness", "The model satisfies every grounded axiom, hence every "
                  "attributed axiom.", "ground attributed Horn DL"),
        Guarantee("strong_faithfulness", "A@S ⊑ B@T holds iff the grounding entails it.",
                  "ground attributed Horn DL"),
    )

    def rejects(self, data):
        from mowl.geometric.attributed import AttributedOntology
        if isinstance(data, AttributedOntology):
            return None
        return (f"it reads an AttributedOntology of mowl.geometric.attributed, not a "
                f"{type(data).__name__}")

    def build(self, data):
        from mowl.geometric.attributed import ground
        from mowl.geometric.completion import CanonicalModel
        from mowl.geometric.normal_forms import NormalisedOntology
        grounding = ground(data)
        normalised = NormalisedOntology(list(grounding.axioms), set(grounding.concept_names),
                                        set(grounding.role_names), 0)
        cm = CanonicalModel.from_normalised(normalised)
        model = FiniteModel.from_canonical_model(cm)
        model.origin = (cm, grounding)
        return model

    def name_of(self, name):
        if isinstance(name, tuple):
            from mowl.geometric.attributed import _name
            return _name(*name)
        return name

    def default_names(self, built):
        from mowl.geometric.attributed import ANY
        cm, grounding = built.origin
        names = [(n, ANY) for n, is_role in sorted(grounding.specifiers) if not is_role]
        return names, sorted(cm.role_names), sorted(cm.individuals)


@register_source
class BoxMILPSource(Source):
    """A box model of a propositional EL⊥ ontology found by the MILP of
    :class:`mowl.geometric.boxes.BoxModelBuilder` after its Helly pre-check. Builds a
    :class:`~mowl.geometric.boxes.BoxModel` (boxes in :math:`[0, M]^d`), not a finite
    model.

    :param d_max, time_limit, faithful, M, delta: Those of ``BoxModelBuilder``.
    """

    key = "box milp"
    kind = "solver"
    logic = "propositional EL⊥ (NF1, NF2 and ⊥ forms; no roles)"
    reasoner = "EL++ completion and a MILP solver (scipy.optimize.milp, HiGHS)"
    guarantees = (
        Guarantee("soundness", "Every returned box model satisfies the normalised axioms "
                  "and has a non-empty box for every satisfiable name.",
                  "propositional EL⊥", "verified on construction"),
        Guarantee("strong_faithfulness", "With faithful='strong' a box inclusion holds iff "
                  "it is entailed (names).", "propositional EL⊥", "faithful='strong'"),
        Guarantee("limitation", "No box model exists if the ontology has a non-degenerate "
                  "Helly certificate (boxes have Helly number 2).", "propositional EL⊥"),
    )

    def __init__(self, d_max=8, time_limit=60.0, faithful="weak", M=100.0, delta=1.0,
                 **options):
        self.d_max, self.time_limit, self.faithful = d_max, time_limit, faithful
        self.M, self.delta, self.options = M, delta, options

    @property
    def produces(self):
        from mowl.geometric.boxes import BoxModel
        return BoxModel

    def rejects(self, data):
        from mowl.geometric.boxes import BoxModel
        return None if isinstance(data, BoxModel) else _rejects_non_ontology(self, data)

    def build(self, data):
        from mowl.geometric.boxes import BoxModel, BoxModelBuilder
        if isinstance(data, BoxModel):
            return data
        cm = canonical_model_of(data)
        return BoxModelBuilder(cm, d_max=self.d_max, time_limit=self.time_limit,
                               faithful=self.faithful, M=self.M, delta=self.delta,
                               **self.options).build()

    def default_names(self, built):
        from mowl.geometric.normal_forms import BOTTOM, TOP, is_nominal_name
        names = sorted(n for n in built.names if n not in (TOP, BOTTOM)
                       and not is_nominal_name(n))
        return names, [], []
