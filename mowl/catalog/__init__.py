"""Catalogue of ontology embedding methods.

Every method known to mOWL, whether it is implemented or only documented, has a
:class:`MethodCard` that records its classification, the logic it covers, its proven
theoretical guarantees and limitations, and where its implementation lives.

.. testcode::

   from mowl.catalog import list_methods, get_method

   for card in list_methods(family="geometric", implemented=True):
       print(card.name, card.region, card.logic)

   card = get_method("elem")
   for g in card.guarantees:
       print(g.kind, g.statement, g.source)

.. testoutput::
   :options: +ELLIPSIS

   ...ELEmbeddings ball...

A publication usually bundles a representation with a way to train it and a way to
evaluate it. mOWL catalogues the three separately: methods (:class:`MethodCard`),
training regimes and evaluation regimes (:class:`RegimeCard`, see
:func:`list_regimes`), so that any regime can be combined with any compatible method.

The catalogue does not need the JVM. Only :meth:`MethodCard.load` (and
:func:`load_method`) import model code.
"""

from mowl.catalog.card import (Family, Status, Origin, GuaranteeKind, Reference, Guarantee,
                               MethodCard, RegimeKind, RegimeCard)
from mowl.catalog.registry import (get_method, list_methods, load_method, get_regime,
                                   list_regimes, regimes_of, card_for)
from mowl.catalog.render import (catalogue_to_rst, card_to_rst, overview_table_rst, parts_table_rst,
                                 regime_to_rst, regimes_to_rst, rst_escape)

__all__ = [
    "Family", "Status", "Origin", "GuaranteeKind", "Reference", "Guarantee", "MethodCard",
    "get_method", "list_methods", "load_method", "card_for",
    "RegimeKind", "RegimeCard", "get_regime", "list_regimes", "regimes_of",
    "catalogue_to_rst", "card_to_rst", "overview_table_rst", "parts_table_rst",
    "regime_to_rst", "regimes_to_rst", "rst_escape",
]
