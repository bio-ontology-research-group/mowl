"""Text from ontologies: entity labels and verbalisation of class expressions
(:mod:`mowl.text.verbalise`). Requires a running JVM."""

from mowl.text.verbalise import (Verbaliser, entity_labels, fallback_label, iri_fragment,
                                 split_camel_case)

__all__ = ["Verbaliser", "entity_labels", "fallback_label", "iri_fragment",
           "split_camel_case"]
