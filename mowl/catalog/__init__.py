"""Catalogue of ontology embedding methods.

Every method known to mOWL, whether it is implemented or only documented, has a
:class:`MethodCard` that records its classification, the logic it covers, its proven
theoretical guarantees and limitations, and where its implementation lives.

.. code-block:: python

   from mowl.catalog import list_methods, get_method

   for card in list_methods(family="geometric", implemented=True):
       print(card.name, card.region, card.logic)

   card = get_method("elem")
   for g in card.guarantees:
       print(g.kind, g.statement, g.source)

The catalogue does not need the JVM. Only :meth:`MethodCard.load` (and
:func:`load_method`) import model code.
"""

from mowl.catalog.card import (Family, Status, Origin, GuaranteeKind, Reference, Guarantee,
                               MethodCard)
from mowl.catalog.registry import get_method, list_methods, load_method
from mowl.catalog.render import catalogue_to_rst, card_to_rst, overview_table_rst

__all__ = [
    "Family", "Status", "Origin", "GuaranteeKind", "Reference", "Guarantee", "MethodCard",
    "get_method", "list_methods", "load_method",
    "catalogue_to_rst", "card_to_rst", "overview_table_rst",
]
