"""The grounded constraints of GroundedEL as a :class:`Program`, for a solver.

GroundedEL's element constraints are rules in the Łukasiewicz logic
(:class:`~mowl.representation.library.grounded_logic.LukasiewiczRules`), and those are
**linear** in the degrees: :math:`A(x) \\wedge B(x) \\to C(x)` is
:math:`m_A(x) + m_B(x) - 1 - m_C(x) \\le 0`. Over a *membership table* (one degree per
element and class, :math:`m \\in [0, 1]^{N \\times C}`, one per pair of elements and relation,
:math:`\\rho \\in [0, 1]^{R \\times N \\times N}`, and the deadness
:math:`\\delta \\in [0, 1]^N`) all grounded constraints together are a linear program, and the
minimality regulariser (:class:`~mowl.representation.library.grounded_logic.Minimality`) is a
linear objective. :class:`GroundedProgram` writes it once against the algebras of
:mod:`~mowl.representation.library.solved_solver`, so it is evaluated with NumPy, fitted by a
solver (:class:`~mowl.representation.library.solved_solver.ScipySolver` on tiny problems) or
by gradient descent (:class:`~mowl.representation.library.solved_solver.GradientFit`).

What is and is not meaningful (the reason for the two readings):

- ``reading="horn"`` (default): every rule holds unconditionally and the bottom rules
  *derive* deadness (:math:`A(x) \\to \\delta_x`, and :math:`r(x, y) \\wedge \\delta_y \\to
  \\delta_x`). These are Horn clauses; in the Łukasiewicz relaxation the least solution is
  crisp and is the least model of the grounded completion rules (a rule whose premises are all
  1 forces its head to 1, and the 0/1 least model satisfies every constraint), so a solver
  with a positive minimality objective returns the least model, the labels of
  :func:`mowl.geometric.least_model.least_model`.
- ``reading="exempt"``: the constraints exactly as GroundedEL's hinges write them: a dead
  element is *exempt* from every rule (:math:`\\dots - \\delta_x \\le 0`), including the
  existence of its witness (:math:`1 - m_A(x_A) - \\delta_{x_A} \\le 0`). Deadness then occurs
  negatively, the clauses are not Horn, and the optimum of the linear program can be
  fractional: for :math:`A \\sqsubseteq \\bot` the existence :math:`m + \\delta \\ge 1` and
  :math:`m \\le \\delta` give :math:`m = \\delta = 1/2`. GroundedEL avoids this in training by
  reading deadness through a crisp gate; a solver cannot. This reading is for evaluating the
  constraints at a trained model (:func:`~mowl.representation.library.solved_solver.evaluate`),
  not for solving.

The *geometry* (positions of the elements, boxes of the classes and relations) is not part of
the program: a union of boxes is a disjunction and the outside of a box (bottom, negative
constraints) is not convex, so the box parameters cannot be fitted by a convex solver. The
solved table is loaded into the geometry by the exact-limit construction
(:meth:`mowl.nn.GroundedELModule.exact_limit_init`: one box per element, at fixed positions),
which realises any crisp table (:func:`load_into`).

Interface summary (for unifying with other constructions)::

    GroundedProgram(table: AxiomTable, top_id, bot_id, reading="horn")
        .variables() -> {"m": (N, C), "rho": (R, N, N), "dead": (N,)}
        .build(alg, v, w) -> objective           # weights eps, eps_r, eps_dead
    crisp_tables(program, values) -> (labels (N, C) bool, edges (E, 3), dead (N,) bool)
    load_into(module, program, values, delta)    # exact-limit construction of the geometry
"""

import numpy as np

from mowl.representation.library.grounded_logic import LukasiewiczRules
from mowl.representation.library.solved_solver import Program

__all__ = ["GroundedProgram", "crisp_tables", "load_into", "READINGS"]

#: The readings of deadness (module docstring).
READINGS = ("horn", "exempt")


class GroundedProgram(Program):
    """The element constraints of GroundedEL over a membership table, a linear program.

    :param table: The axioms (:class:`~mowl.representation.library.solved_grounding.AxiomTable`)
        over a domain of ``nb_classes`` class elements and ``nb_individuals`` individuals.
    :param top_id, bot_id: Class ids of top and bottom.
    :param reading: ``"horn"`` (solvable; default) or ``"exempt"`` (GroundedEL's hinges).
    :param rules: The rules (must be linear: the Łukasiewicz rules).
    """

    weights = ("eps", "eps_r", "eps_dead")

    def __init__(self, table, top_id=None, bot_id=None, reading="horn", rules=None):
        if reading not in READINGS:
            raise ValueError(f"reading must be one of {READINGS}, got {reading!r}.")
        self.table, self.top_id, self.bot_id, self.reading = table, top_id, bot_id, reading
        self.rules = rules if rules is not None else LukasiewiczRules()
        if not hasattr(self.rules, "constraint"):
            raise TypeError("Only rules that are linear constraints (Łukasiewicz) make a "
                            "linear program.")
        d = table.domain
        self.nb_classes, self.nb_rels, self.size = d.nb_classes, d.nb_relations, d.size

    def variables(self):
        n = self.size
        return {"m": (n, self.nb_classes), "rho": (self.nb_rels, n, n), "dead": (n,)}

    def _rows(self, form):
        rows = self.table.instances(form)
        if not len(rows):
            return []
        width = len(self.table.COLUMNS[form])
        return [tuple(int(v) for v in row[:width]) for row in np.asarray(rows)]

    def build(self, alg, v, w):
        m, rho, dead = v["m"], v["rho"], v["dead"]
        rule = self.rules.constraint
        exempt = self.reading == "exempt"
        ex = dead if exempt else None              # the exemption of an element rule
        nc = self.nb_classes
        # degrees in [0, 1]
        for name in ("m", "rho", "dead"):
            x = v[name]
            if x is not None:
                alg.le(0.0 * x, x, f"{name} >= 0")
                alg.le(x, 0.0 * x + 1.0, f"{name} <= 1")
        # the witness of every class exists; top holds everywhere, bottom nowhere
        for c in range(nc):
            if c == self.bot_id:
                continue
            rule(alg, [], m[c, c], dead[c] if exempt else None, "existence")
        if self.top_id is not None:
            rule(alg, [], m[:, self.top_id], ex, "top")
        if self.bot_id is not None:
            rule(alg, [m[:, self.bot_id]], None, None, "bottom")
            rule(alg, [], dead[self.bot_id], None, "bottom is dead")
        for a, b in self._rows("gci0"):
            rule(alg, [m[:, a]], m[:, b], ex, "gci0")
        for a, b, c in self._rows("gci1"):
            rule(alg, [m[:, a], m[:, b]], m[:, c], ex, "gci1")
        for a, r, b in self._rows("gci2"):
            rule(alg, [m[:, a]], rho[r][:, b], ex, "gci2")
        for r, b, c in self._rows("gci3"):
            rule(alg, [rho[r], m[:, b][None, :]], m[:, c][:, None],
                 ex[:, None] if exempt else None, "gci3")
        for (a,) in self._rows("gci0_bot"):
            rule(alg, [m[:, a]], dead, None, "gci0_bot")
        for a, b in self._rows("gci1_bot"):
            rule(alg, [m[:, a], m[:, b]], dead, None, "gci1_bot")
        for r, b in self._rows("gci3_bot"):
            rule(alg, [rho[r], m[:, b][None, :]], dead[:, None], None, "gci3_bot")
        for r in range(self.nb_rels):              # a relation to a dead element kills
            rule(alg, [rho[r], dead[None, :]], dead[:, None], None, "dead propagation")
        for r, s in self._rows("role_inclusion"):
            rule(alg, [rho[r]], rho[s], None, "role_inclusion")
        for r, s, t in self._rows("role_chain"):
            rule(alg, [rho[r][:, :, None], rho[s][None, :, :]], rho[t][:, None, :], None,
                 "role_chain")
        for a, c in self._rows("class_assertion"):
            if c == self.bot_id:
                rule(alg, [], dead[nc + a], None, "class_assertion")
            else:
                rule(alg, [], m[nc + a, c], dead[nc + a] if exempt else None,
                     "class_assertion")
        for a, r, b in self._rows("object_property_assertion"):
            rule(alg, [], rho[r][nc + a, nc + b], None, "object_property_assertion")
        # the minimality regulariser: means of the degrees
        n = self.size
        objective = w["eps"] * alg.sum(alg.flatten(m)) / float(n * nc) \
            + w["eps_dead"] * alg.sum(dead) / float(n)
        if self.nb_rels:
            objective = objective + w["eps_r"] * alg.sum(alg.flatten(rho)) / float(
                self.nb_rels * n * n)
        return objective


def crisp_tables(program, values, threshold=0.5):
    """The crisp model of a solution: ``(labels (N, C) bool, edges (E, 3) [x, y, r],
    dead (N,) bool)``, with the rows of dead elements emptied and no edge at a dead element
    (the pruned view of :class:`mowl.geometric.least_model.LeastModel`)."""
    m = np.asarray(values["m"]) >= threshold
    dead = np.asarray(values["dead"]) >= threshold
    m[dead] = False
    if program.bot_id is not None:
        m[:, program.bot_id] = False
    rho = np.asarray(values["rho"]) >= threshold
    edges = [(x, y, r) for r in range(rho.shape[0]) for x in range(rho.shape[1])
             for y in range(rho.shape[2]) if rho[r, x, y] and not dead[x] and not dead[y]]
    return m, np.asarray(edges, dtype=np.int64).reshape(-1, 3), dead


def load_into(module, program, values, delta=0.05, seed=0):
    """Load a solution into the geometry: the exact-limit construction of the module (one
    box of half-width ``delta`` per element and true label, one per true edge) realises the
    crisp table; needs ``module.n_components >= module.nb_elements``."""
    import torch as th
    labels, edges, dead = crisp_tables(program, values)
    module.exact_limit_init(th.as_tensor(labels), edges=th.as_tensor(edges), delta=delta,
                            seed=seed, dead_mask=th.as_tensor(dead))
    return labels, edges, dead
