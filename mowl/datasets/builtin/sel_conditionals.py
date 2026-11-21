"""Statistical EL ontologies of probabilistic conditionals, for BoxSEL.

BoxSEL (Zhu et al., 2024, arXiv:2407.11821) is evaluated on three Statistical EL ontologies
(Person, Country, Hybrid) that the authors derived from YAGO3 by counting; the ontologies
are **not released** (the paper links no code or data), and the selection of concepts and
roles of the Person and Country ontologies is only partly stated. This module provides what
the paper does state, so that the ontologies can be rebuilt from any knowledge base with
complete class assertions:

* :func:`conditionals_from_assertions` is Algorithm 1 of the paper (Appendix "Details for SEL
  ontology construction"): all conditionals ``(B|A)``, ``(B|A1 ⊓ A2)``, ``(B|∃r.A)`` and
  ``(∃r.A|B)`` with their statistical proportions;
* :class:`SELConditionalDataset` reads such a set of conditionals from a JSON file, splits it
  for the probabilistic modus ponens evaluation of the paper and exposes the training
  conditionals and the queries.

The file format is mOWL's own: ``{"root": iri, "conditionals": [{"conclusion": e, "premise":
e, "lower": l, "upper": u}, ...]}`` where an expression ``e`` is a class IRI,
``["and", e1, e2]`` or ``["some", role_iri, e]`` (the expressions of
:class:`~mowl.models.boxsel.model.BoxSEL`).
"""

import hashlib
import itertools
import json
import os
import random

from mowl.datasets.base import PathDataset, default_data_root

IRI_PREFIX = "http://mowl.borg/sel/"


def _key(expr):
    """A hashable normal form of a conditional expression (conjuncts are unordered)."""
    if isinstance(expr, str):
        return expr
    if expr[0] == "and":
        parts = []
        for e in expr[1:]:
            k = _key(e)
            parts.extend(k[1:] if isinstance(k, tuple) and k[0] == "and" else [k])
        return ("and",) + tuple(sorted(parts, key=repr))
    if expr[0] == "some":
        return ("some", expr[1], _key(expr[2]))
    raise ValueError(f"Unknown expression {expr!r}.")


def _expression(key):
    """Back from a key to the list form of an expression."""
    if isinstance(key, str):
        return key
    if key[0] == "and":
        return ["and"] + [_expression(k) for k in key[1:]]
    return ["some", key[1], _expression(key[2])]


def _names(expr, classes, roles):
    if isinstance(expr, str):
        classes.add(expr)
    elif expr[0] == "and":
        for e in expr[1:]:
            _names(e, classes, roles)
    else:
        roles.add(expr[1])
        _names(expr[2], classes, roles)


def conditionals_from_assertions(members, role_pairs=None):
    """Algorithm 1 of the paper: the conditionals PNF1-PNF4 of a knowledge base.

    :param members: ``{class IRI: set of entities}`` (complete class assertions: an entity of
        a subclass is also a member of the superclass).
    :param role_pairs: ``{role IRI: set of (entity, entity)}``.
    :return: A list of ``(conclusion, premise, p)`` with expressions in the list form of
        the file format. Conditionals with an empty premise are undefined and skipped.
        ``(B | A1 ⊓ A2)`` is generated once per unordered pair ``{A1, A2}`` (the paper's
        loop visits both orders, which gives the same conditional twice).
    """
    role_pairs = role_pairs or {}
    classes = sorted(members)
    out = []
    for a in classes:
        na = len(members[a])
        if not na:
            continue
        for b in classes:
            if b != a:
                out.append((b, a, len(members[a] & members[b]) / na))
    for a1, a2 in itertools.combinations(classes, 2):
        both = members[a1] & members[a2]
        if not both:
            continue
        for b in classes:
            if b not in (a1, a2):
                out.append((b, ["and", a1, a2], len(both & members[b]) / len(both)))
    for r in sorted(role_pairs):
        pairs = role_pairs[r]
        for a in classes:
            # entities with an r-successor in A
            having = {x for x, y in pairs if y in members[a]}
            for b in classes:
                if b == a:
                    continue
                joint = having & members[b]
                if having:
                    out.append((b, ["some", r, a], len(joint) / len(having)))
                if members[b]:
                    out.append((["some", r, a], b, len(joint) / len(members[b])))
    return out


def write_conditionals_file(path, conditionals, root):
    """Write conditionals ``(conclusion, premise, p)`` or ``(conclusion, premise, l, u)`` in
    the file format of :class:`SELConditionalDataset`, atomically."""
    items = []
    for c in conditionals:
        lower, upper = (c[2], c[2]) if len(c) == 3 else (c[2], c[3])
        items.append({"conclusion": c[0], "premise": c[1], "lower": lower, "upper": upper})
    partial = f"{path}.{os.getpid()}.part"
    with open(partial, "w", encoding="utf-8") as f:
        json.dump({"root": root, "conditionals": items}, f)
    os.replace(partial, path)


class SELConditionalDataset(PathDataset):
    """A set of probabilistic conditionals with the split of the BoxSEL evaluation.

    The conditionals whose premise is the *root* concept (the most general concept of the
    knowledge base, ``D'`` of the paper) are the candidate queries; a random ``query_fraction``
    (paper: 30%) of them is held out as the query set and the other conditionals are the
    learning set ``D_model`` that the embedding is trained on (Appendix "Pseudocode for the
    PMP evaluation"). The split depends on ``split_seed`` (the paper varies it over five
    query sets).

    The ontology of the dataset has no axioms, only the declarations of the classes and roles
    of the conditionals: everything the model learns is in :attr:`conditionals`.

    The file is not downloaded: the ontologies of the paper are not released. Give the path
    of a file you built (see :func:`conditionals_from_assertions`), or put
    ``<name>.json`` into ``<data_root>/sel_conditionals/``.

    :param name: Name of the ontology (``"person"``, ``"country"``, ``"hybrid"`` in the
        paper's experiments); the file is ``<data_root>/sel_conditionals/<name>.json``.
    :param path: Alternatively, the path of the file.
    :param query_fraction: Fraction of the candidate queries that is held out.
    :param split_seed: Seed of the split.
    :param data_root: Directory of the files (default: :func:`default_data_root`).
    """

    def __init__(self, name="person", path=None, query_fraction=0.3, split_seed=0,
                 data_root=None):
        if not 0.0 < query_fraction < 1.0:
            raise ValueError("query_fraction must be in (0, 1).")
        root_dir = default_data_root() if data_root is None else data_root
        self.directory = os.path.join(root_dir, "sel_conditionals")
        self.path = path or os.path.join(self.directory, f"{name}.json")
        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"{self.path} does not exist. The Statistical EL ontologies of the BoxSEL "
                f"paper (Person, Country, Hybrid, derived from YAGO3) are not released. Build "
                f"one with mowl.datasets.builtin.sel_conditionals.conditionals_from_assertions "
                f"and write_conditionals_file, and pass its path or place it there.")
        self.name, self.query_fraction, self.split_seed = name, query_fraction, split_seed
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        self.root = data["root"]
        everything = [(c["conclusion"], c["premise"], float(c["lower"]), float(c["upper"]))
                      for c in data["conditionals"]]
        candidates = [i for i, c in enumerate(everything) if _key(c[1]) == _key(self.root)]
        rng = random.Random(split_seed)
        held = set(rng.sample(candidates, int(round(query_fraction * len(candidates)))))
        self.all_conditionals = everything
        self.conditionals = [c for i, c in enumerate(everything) if i not in held]
        self.queries = [c for i, c in enumerate(everything) if i in held]
        classes, roles = set(), set()
        for c in everything:
            _names(c[0], classes, roles)
            _names(c[1], classes, roles)
        if isinstance(self.root, str):
            classes.add(self.root)
        digest = hashlib.sha1(os.path.abspath(self.path).encode()).hexdigest()[:10]
        declared = os.path.join(root_dir, "sel_conditionals", "declarations",
                                f"{name}_{digest}.ofn")
        self._write_declarations(declared, sorted(classes), sorted(roles))
        super().__init__(declared)

    @staticmethod
    def _write_declarations(path, classes, roles):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        partial = f"{path}.{os.getpid()}.part"
        with open(partial, "w", encoding="utf-8") as out:
            out.write(f"Ontology(<{IRI_PREFIX}{os.path.basename(path)}>\n")
            for c in classes:
                out.write(f"Declaration(Class(<{c}>))\n")
            for r in roles:
                out.write(f"Declaration(ObjectProperty(<{r}>))\n")
            out.write(")\n")
        os.replace(partial, path)

    # -- the probabilistic modus ponens queries ------------------------------------------
    def pmp_queries(self):
        """The evaluable queries and their entailed intervals.

        A query ``(E | C)`` of the query set is evaluable if the learning set has
        premises ``(D | C)[l1, u1]`` and ``(E | C ⊓ D)[l2, u2]`` (Proposition 1, the
        probabilistic modus ponens); then ``(E | C)`` lies in
        ``[l1 l2, min(1, u1 u2 + 1 - l1)]``. With several such ``D`` the entailed interval is
        the intersection of the intervals (all are consequences of the learning set).

        :return: A list of ``(conclusion, premise, lower, upper)`` with the expressions of
            the query and the entailed interval; ones whose intersection is empty are
            dropped.
        """
        by_premise = {}
        index = {}
        for e, c, lo, up in self.conditionals:
            index[(_key(e), _key(c))] = (lo, up)
            by_premise.setdefault(_key(c), []).append((e, lo, up))
        out = []
        for e, c, _, _ in self.queries:
            lower, upper, found = 0.0, 1.0, False
            ck = _key(c)
            for d, l1, u1 in by_premise.get(ck, ()):
                body = _key(["and", _expression(ck), d])
                second = index.get((_key(e), body))
                if second is None:
                    continue
                l2, u2 = second
                lower, upper, found = (max(lower, l1 * l2), min(upper, min(1.0, u1 * u2 + 1 - l1)),
                                       True)
            if found and lower <= upper:
                out.append((e, c, lower, upper))
        return out
