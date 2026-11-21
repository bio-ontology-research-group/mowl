"""Registry of :class:`~mowl.catalog.card.MethodCard` and
:class:`~mowl.catalog.card.RegimeCard` objects.

Regime cards live in :mod:`mowl.catalog.regimes`, one module per regime or group of
regimes, exposing ``REGIME`` or ``REGIMES``.

Cards live in the modules of :mod:`mowl.catalog.entries`, one module per method (or per
closely related group of methods). A module exposes its cards as a module-level ``CARD``
(one card) or ``CARDS`` (an iterable of cards). All modules of the package are discovered
automatically, so adding a method to the catalogue means adding one file.
"""

import importlib
import pkgutil

from mowl.catalog.card import MethodCard, Family, Status, RegimeCard, RegimeKind

_CARDS = None
_REGIMES = None


def _discover():
    from mowl.catalog import entries

    cards = {}
    for info in sorted(pkgutil.iter_modules(entries.__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{entries.__name__}.{info.name}")
        found = []
        if hasattr(module, "CARD"):
            found.append(module.CARD)
        found.extend(getattr(module, "CARDS", ()))
        if not found:
            raise ImportError(
                f"{module.__name__} defines neither CARD nor CARDS.")
        for card in found:
            if not isinstance(card, MethodCard):
                raise TypeError(f"{module.__name__} exposes a non-MethodCard object.")
            if card.key in cards:
                raise ValueError(f"Duplicate catalogue key {card.key!r}.")
            cards[card.key] = card
    return cards


def _cards():
    global _CARDS
    if _CARDS is None:
        _CARDS = _discover()
    return _CARDS


def get_method(key):
    """Return the card registered under ``key``.

    :raises KeyError: If no such method is catalogued.
    """
    cards = _cards()
    if key not in cards:
        raise KeyError(f"Unknown method {key!r}. Known methods: {sorted(cards)}")
    return cards[key]


def list_methods(family=None, status=None, logic=None, implemented=None, task=None,
                 covers=None):
    """Return catalogued methods, optionally filtered, sorted by family and key.

    :param family: Keep only this :class:`~mowl.catalog.card.Family` (or its value).
    :param status: Keep only this :class:`~mowl.catalog.card.Status` (or its value).
    :param logic: Keep only cards whose ``logic`` contains this string (case-insensitive).
    :param implemented: If ``True``/``False``, keep only implemented/unimplemented methods.
    :param task: Keep only methods that give a score for this inference task (a key of
        :data:`mowl.logic.TASKS`).
    :param covers: Keep only methods whose supported expressivity includes this one (a
        :class:`mowl.logic.Expressivity` or its name), e.g. the expressivity of an ontology
        (:func:`mowl.logic.ontology_expressivity`).
    :rtype: list[MethodCard]
    """
    family = Family(family) if family is not None else None
    status = Status(status) if status is not None else None
    if isinstance(covers, str):
        from mowl.logic import Expressivity
        covers = Expressivity.parse(covers)
    out = []
    for card in _cards().values():
        if family is not None and card.family is not family:
            continue
        if status is not None and card.status is not status:
            continue
        if logic is not None and logic.lower() not in card.logic.lower():
            continue
        if implemented is not None and card.implemented != implemented:
            continue
        if task is not None and task not in card.tasks:
            continue
        if covers is not None:
            supported = card.supported_expressivity
            if supported is None or not covers <= supported:
                continue
        out.append(card)
    return sorted(out, key=lambda c: (c.family.value, c.region, c.key))


def card_for(cls):
    """Return the card of the method implemented by the class ``cls`` (or by the class of
    an instance), looking along its base classes; ``None`` if it has no card."""
    if not isinstance(cls, type):
        cls = type(cls)
    by_path = {c.implementation: c for c in _cards().values() if c.implementation}
    for base in cls.__mro__:
        card = by_path.get(f"{base.__module__}.{base.__qualname__}")
        if card is not None:
            return card
    return None


def load_method(key):
    """Import and return the class implementing the method ``key`` (needs the JVM)."""
    return get_method(key).load()


def _discover_regimes():
    from mowl.catalog import regimes

    found = {}
    for info in sorted(pkgutil.iter_modules(regimes.__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{regimes.__name__}.{info.name}")
        items = list(getattr(module, "REGIMES", ()))
        if hasattr(module, "REGIME"):
            items.append(module.REGIME)
        if not items:
            raise ImportError(f"{module.__name__} defines neither REGIME nor REGIMES.")
        for card in items:
            if not isinstance(card, RegimeCard):
                raise TypeError(f"{module.__name__} exposes a non-RegimeCard object.")
            if card.key in found:
                raise ValueError(f"Duplicate regime key {card.key!r}.")
            found[card.key] = card
    return found


def _regimes():
    global _REGIMES
    if _REGIMES is None:
        _REGIMES = _discover_regimes()
    return _REGIMES


def get_regime(key):
    """Return the regime card registered under ``key``."""
    regimes = _regimes()
    if key not in regimes:
        raise KeyError(f"Unknown regime {key!r}. Known regimes: {sorted(regimes)}")
    return regimes[key]


def list_regimes(kind=None, implemented=None, method=None):
    """Return catalogued regimes, optionally filtered, sorted by kind and key.

    :param kind: Keep only this :class:`~mowl.catalog.card.RegimeKind` (or its value).
    :param implemented: If ``True``/``False``, keep only implemented/unimplemented ones.
    :param method: Keep only regimes introduced with or used by this method key.
    :rtype: list[RegimeCard]
    """
    kind = RegimeKind(kind) if kind is not None else None
    out = [c for c in _regimes().values()
           if (kind is None or c.kind is kind)
           and (implemented is None or c.implemented == implemented)
           and (method is None or method in c.introduced_with or method in c.used_by)]
    return sorted(out, key=lambda c: (c.kind.value, c.key))


def regimes_of(method, kind=None):
    """Return the keys of the regimes linked to the method ``method``.

    The link is recorded on either side: a method card names regimes in ``training`` and
    ``evaluation``, and a regime card names methods in ``introduced_with`` and ``used_by``.
    This function returns the union, the method card's own entries first.

    :param method: A catalogue key.
    :param kind: Keep only this :class:`~mowl.catalog.card.RegimeKind` (or its value).
    :rtype: list[str]
    """
    kind = RegimeKind(kind) if kind is not None else None
    card = get_method(method)
    keys = []
    if kind is None or kind is RegimeKind.TRAINING:
        keys.extend(card.training)
    if kind is None or kind is RegimeKind.EVALUATION:
        keys.extend(card.evaluation)
    keys.extend(c.key for c in list_regimes(kind=kind, method=method))
    return list(dict.fromkeys(keys))
