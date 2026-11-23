"""The module classes of the methods that are combinations of parts.

``mowl.nn.ELBEModule`` and its siblings (``BoxELModule``, ``TransBoxModule``, ...) are
:class:`~mowl.representation.ComposedGeometry` objects: the method is the combination of a
class model, a relation model, a measure and an objective that its builder
(:func:`mowl.representation.register_method`) defines, and the module class only keeps the
constructor, the attribute names and the hooks that the released class had.
:class:`_PartBuiltModule` is what these classes share.
"""


import torch as th

from mowl.nn.el.elmodule import ELModule
from mowl.representation.geometry import ComposedGeometry

__all__ = []


class _PartBuiltModule(ComposedGeometry):
    """A :class:`~mowl.representation.ComposedGeometry` whose parts come from the builder of
    a published method.

    * The per-axiom losses are looked up by *method* (``gci0_loss``, ...), like those of every
      :class:`~mowl.nn.ELModule`, so a subclass that overrides one of them is called
      (:class:`~mowl.representation.ComposedGeometry` itself asks the objective directly).
    * ``strict_negatives`` is ``True`` as in every module: a negative loss of a form without
      one is an error.
    * Checkpoints saved by the earlier, hand-written module of the method still load:
      :attr:`state_dict_aliases` names the earlier parameter names.

    :param built: The geometry returned by the builder of the method.
    :param individuals: Replaces the individual model of ``built`` (to keep an empty table
        where the earlier module had one).
    """

    #: ``{earlier parameter name: name in the part-built module}``, for checkpoints of the
    #: hand-written module of the method (``{"min_embedding.weight":
    #: "class_model.mins.weight"}``).
    state_dict_aliases = {}

    #: Whether the state dict also names every table by its name in the released module
    #: (:attr:`state_dict_aliases`), so that code reading ``state_dict()["min_embedding.weight"]``
    #: keeps working (the module classes released in mOWL 2.1.0 set it).
    released_state_dict_names = False

    #: The forms whose loss the hand-written module returned as a flat vector ``(B,)``: with
    #: ``per_axiom_column = False`` they still do (see
    #: :class:`~mowl.nn.el.per_axiom_column.PerAxiomColumn`); by default every loss is a column.
    flat_forms = frozenset()

    def __init__(self, built, individuals=None):
        ComposedGeometry.__init__(
            self, built.class_model, built.relation_model, built.measure, built.objective,
            individuals=individuals if individuals is not None else built.individual_model,
            regularizers=built.regularizers, supported_forms=built.supported_forms,
            neg_capable_gcis=built.neg_capable_gcis)
        self.strict_negatives = True
        # A class that declares its negative losses (``neg_capable_gcis`` on the class, as
        # the released modules do and a subclass may change) decides, not the objective.
        for cls in type(self).__mro__:
            if cls is ComposedGeometry:
                break
            if "neg_capable_gcis" in vars(cls):
                del self.neg_capable_gcis
                break
        self._register_load_state_dict_pre_hook(_rename_earlier_parameters, with_module=True)
        if self.released_state_dict_names:
            self._register_released_names()

    def _register_released_names(self):
        """Register every table also under its released name (one more name of the same
        table, as :attr:`state_dict_aliases` lists them)."""
        for old, new in self.state_dict_aliases.items():
            name, path = old.rsplit(".", 1)[0], new.rsplit(".", 1)[0]
            try:
                table = self.get_submodule(path)
            except AttributeError:
                continue                    # (no individuals)
            if name not in self._modules:
                th.nn.Module.__setattr__(self, name, table)

    def get_loss_function(self, gci_name):
        return ELModule.get_loss_function(self, gci_name)

    def cost(self, form, rows, neg=False):
        out = super().cost(form, rows, neg)
        if not self.per_axiom_column and form in self.flat_forms and out.ndim == 2 \
                and out.shape[1] == 1:
            out = out.reshape(-1)
        return out


def _rename_earlier_parameters(module, state_dict, prefix, *args):
    """Load hook: a checkpoint of the hand-written module has its parameters under the
    earlier names; give them the names of the parts (and of every alias of the same
    table) and drop the earlier names."""
    own = module.state_dict(keep_vars=True)
    for old, new in module.state_dict_aliases.items():
        if prefix + old not in state_dict or new not in own:
            continue
        value = state_dict[prefix + old]
        for name, tensor in own.items():
            if tensor is own[new] and prefix + name not in state_dict:
                state_dict[prefix + name] = value
        if old not in own:
            del state_dict[prefix + old]
