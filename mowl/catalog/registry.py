"""Registry of :class:`~mowl.catalog.card.MethodCard` objects.

Cards live in the modules of :mod:`mowl.catalog.entries`, one module per method (or per
closely related group of methods). A module exposes its cards as a module-level ``CARD``
(one card) or ``CARDS`` (an iterable of cards). All modules of the package are discovered
automatically, so adding a method to the catalogue means adding one file.
"""

import importlib
import pkgutil

from mowl.catalog.card import MethodCard, Family, Status

_CARDS = None


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


def list_methods(family=None, status=None, logic=None, implemented=None):
    """Return catalogued methods, optionally filtered, sorted by family and key.

    :param family: Keep only this :class:`~mowl.catalog.card.Family` (or its value).
    :param status: Keep only this :class:`~mowl.catalog.card.Status` (or its value).
    :param logic: Keep only cards whose ``logic`` contains this string (case-insensitive).
    :param implemented: If ``True``/``False``, keep only implemented/unimplemented methods.
    :rtype: list[MethodCard]
    """
    family = Family(family) if family is not None else None
    status = Status(status) if status is not None else None
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
        out.append(card)
    return sorted(out, key=lambda c: (c.family.value, c.region, c.key))


def load_method(key):
    """Import and return the class implementing the method ``key`` (needs the JVM)."""
    return get_method(key).load()
