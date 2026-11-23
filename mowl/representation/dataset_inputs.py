"""What a method takes from the dataset besides the sizes of its signature.

Most methods are built from sizes alone (numbers of classes, relations and individuals, and
a dimension), so a general model (:class:`~mowl.models.ComposedEL`) can build any of them and
swap their parts. A method whose geometry has a part that is *constructed from the
ontology* before training -- FaithEL's strongly faithful TBox model, the domain of a grounded
model -- also needs that part. The method registers a function here that makes it from the
model (its dataset and index maps): :func:`dataset_inputs` returns the options to pass to
the builder of the method, so the general model builds the method like the named class
does. ``embed_dim`` among them is the dimension the construction fixes (the general model
takes it instead of its own).
"""

__all__ = ["DATASET_INPUTS", "register_dataset_inputs", "dataset_inputs"]

#: key of a method -> ``function(model) -> {option: value}``
DATASET_INPUTS = {}


def register_dataset_inputs(key):
    """Register the function that makes the constructed inputs of the method ``key``."""
    def wrap(function):
        if key in DATASET_INPUTS and DATASET_INPUTS[key] is not function:
            raise ValueError(f"Dataset inputs of {key!r} are registered already.")
        DATASET_INPUTS[key] = function
        return function
    return wrap


def dataset_inputs(key, model):
    """The options the builder of ``key`` takes from ``model`` (its dataset and index maps);
    empty for a method built from sizes alone."""
    function = DATASET_INPUTS.get(key)
    return {} if function is None else dict(function(model))
