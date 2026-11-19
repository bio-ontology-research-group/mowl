"""Render the method catalogue as reStructuredText.

Used by ``docs/source/conf.py`` to generate ``docs/source/methods/catalogue.rst`` at
documentation build time, so the documentation can never drift from the cards.
"""

from mowl.catalog.card import Family, Status, RegimeKind
from mowl.catalog.registry import list_methods, list_regimes, regimes_of

FAMILY_TITLES = {
    Family.GEOMETRIC: "Geometric (region-based) methods",
    Family.CONSTRUCTIVE: "Constructive exact models",
    Family.FUZZY: "Fuzzy and model-sampling methods",
    Family.ORDER: "Order and lattice embeddings",
    Family.ALGEBRAIC: "Algebraic methods (subspaces, distributions)",
    Family.LANGUAGE_MODEL: "Language-model based methods",
    Family.GRAPH: "Graph-based methods",
    Family.SYNTACTIC: "Syntactic (corpus-based) methods",
}

STATUS_LABELS = {
    Status.LEARNABLE: "trainable",
    Status.CONSTRUCTIVE: "construction",
    Status.CATALOGUED: "documented only",
}


def _esc(text):
    # A bare "|" starts a substitution reference in reStructuredText.
    return " ".join(str(text).split()).replace("|", "\\|")


def _title(text, char):
    return f"{text}\n{char * len(text)}\n"


def card_to_rst(card):
    """Return the reStructuredText section describing one card."""
    lines = [f".. _method-{card.key}:\n", _title(card.name, "^")]
    lines.append(_esc(card.summary) + "\n")
    rows = [
        ("Status", STATUS_LABELS[card.status]
         + (f" ({card.origin.value})" if card.implemented else "")),
        ("Region / type", card.region),
        ("Logic", card.logic),
        ("Concepts", card.concepts),
        ("Roles", card.roles),
    ]
    if card.individuals:
        rows.append(("Individuals", card.individuals))
    if card.implemented:
        rows.append(("Implementation", f":class:`{card.implementation}`"))
        if card.axioms:
            rows.append(("Axioms trained on", ", ".join(f"``{a}``" for a in card.axioms)))
    else:
        rows.append(("Why not implemented", card.not_implemented_reason))
    if card.intersection_closed is not None:
        closed = card.intersection_closed
        rows.append(("Closed under intersection",
                     closed if isinstance(closed, str) else ("yes" if closed else "no")))
    if card.helly:
        rows.append(("Helly number", card.helly))
    if card.related:
        rows.append(("Related methods",
                     ", ".join(f":ref:`{k} <method-{k}>`" for k in card.related)))
    if card.optional_dependencies:
        rows.append(("Optional dependencies", ", ".join(card.optional_dependencies)))
    if card.code_url:
        rows.append(("Reference code", f"`{card.code_url} <{card.code_url}>`__"))
    training = regimes_of(card.key, RegimeKind.TRAINING)
    if training:
        rows.append(("Training regimes (paper)",
                     ", ".join(f":ref:`{k} <regime-{k}>`" for k in training)))
    evaluation = regimes_of(card.key, RegimeKind.EVALUATION)
    if evaluation:
        rows.append(("Evaluation regimes (paper)",
                     ", ".join(f":ref:`{k} <regime-{k}>`" for k in evaluation)))
    if card.benchmarks:
        rows.append(("Published results recorded for",
                     ", ".join(f"``{b}``" for b in card.benchmarks)))
    lines.append(".. list-table::\n   :widths: 25 75\n")
    for k, v in rows:
        lines.append(f"   * - {k}\n     - {_esc(v)}")
    lines.append("")
    if card.guarantees:
        lines.append("**Theoretical statements**\n")
        for g in card.guarantees:
            verified = f" Checked in mOWL by ``{g.verified_in_mowl}``." if g.verified_in_mowl \
                else ""
            lines.append(f"* *{g.kind.value.replace('_', ' ')}* ({_esc(g.condition)}): "
                         f"{_esc(g.statement)} [{_esc(g.source)}]{verified}")
        lines.append("")
    if card.limitations:
        lines.append("**Known limitations**\n")
        lines.extend(f"* {_esc(text)}" for text in card.limitations)
        lines.append("")
    if card.notes:
        lines.append(f"**Notes.** {_esc(card.notes)}\n")
    lines.append("**References**\n")
    for r in card.references:
        url = f" `{r.url} <{r.url}>`__" if r.url else ""
        lines.append(f"* {_esc(r.citation)}{url}")
    lines.append("")
    return "\n".join(lines)


def overview_table_rst(cards=None):
    """Return a ``list-table`` with one row per method."""
    cards = list_methods() if cards is None else cards
    lines = [".. list-table::", "   :header-rows: 1", "   :widths: 16 14 14 14 14 28", "",
             "   * - Method", "     - Family", "     - Region / type", "     - Logic",
             "     - In mOWL", "     - Strongest proven statement"]
    for c in cards:
        best = ""
        for g in c.guarantees:
            if g.kind.value != "limitation":
                best = f"{g.kind.value.replace('_', ' ')} ({_esc(g.condition)})"
                break
        lines += [f"   * - :ref:`{c.name} <method-{c.key}>`", f"     - {c.family.value}",
                  f"     - {_esc(c.region)}", f"     - {_esc(c.logic)}",
                  f"     - {STATUS_LABELS[c.status]}", f"     - {best or '—'}"]
    return "\n".join(lines) + "\n"


def catalogue_to_rst():
    """Return the full catalogue page."""
    out = [".. This file is generated from mowl.catalog by docs/source/conf.py. "
           "Do not edit.\n", _title("Method catalogue", "="),
           "Overview\n--------\n", overview_table_rst()]
    for family, title in FAMILY_TITLES.items():
        cards = list_methods(family=family)
        if not cards:
            continue
        out.append(_title(title, "-"))
        out.extend(card_to_rst(c) for c in cards)
    return "\n".join(out)


def regime_to_rst(card):
    """Return the reStructuredText section describing one regime card."""
    lines = [f".. _regime-{card.key}:\n", _title(card.name, "^"), _esc(card.summary) + "\n"]
    rows = [("Key", f"``{card.key}``"),
            ("Determines", ", ".join(card.components)),
            ("Applies to", card.applies_to),
            ("Published", "as a contribution of its own" if card.standalone
             else "as part of a method publication")]
    if card.introduced_with:
        rows.append(("Introduced with",
                     ", ".join(f":ref:`{k} <method-{k}>`" for k in card.introduced_with)))
    if card.used_by:
        rows.append(("Also used by",
                     ", ".join(f":ref:`{k} <method-{k}>`" for k in card.used_by)))
    if card.implemented:
        rows.append(("Implementation", f":class:`{card.implementation}`"))
    else:
        rows.append(("Why not implemented", card.not_implemented_reason))
    lines.append(".. list-table::\n   :widths: 25 75\n")
    for k, v in rows:
        lines.append(f"   * - {k}\n     - {_esc(v)}")
    lines.append("")
    if card.properties:
        lines.append("**Properties**\n")
        for g in card.properties:
            lines.append(f"* ({_esc(g.condition)}) {_esc(g.statement)} [{_esc(g.source)}]")
        lines.append("")
    if card.limitations:
        lines.append("**Known limitations**\n")
        lines.extend(f"* {_esc(t)}" for t in card.limitations)
        lines.append("")
    if card.notes:
        lines.append(f"**Notes.** {_esc(card.notes)}\n")
    lines.append("**References**\n")
    for r in card.references:
        url = f" `{r.url} <{r.url}>`__" if r.url else ""
        lines.append(f"* {_esc(r.citation)}{url}")
    lines.append("")
    return "\n".join(lines)


def regimes_to_rst():
    """Return the full page of training and evaluation regimes."""
    out = [".. This file is generated from mowl.catalog by docs/source/conf.py. "
           "Do not edit.\n", _title("Training and evaluation regimes", "=")]
    for kind, title in ((RegimeKind.TRAINING, "Training regimes"),
                        (RegimeKind.EVALUATION, "Evaluation regimes")):
        cards = list_regimes(kind=kind)
        if not cards:
            continue
        out.append(_title(title, "-"))
        out.extend(regime_to_rst(c) for c in cards)
    return "\n".join(out)
