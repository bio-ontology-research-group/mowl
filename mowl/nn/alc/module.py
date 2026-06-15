import torch.nn as nn


class ALCModule(nn.Module):
    r"""Subclass of :class:`torch.nn.Module` for :math:`\mathcal{ALC}` models.

    This class provides an interface for neural modules that embed the
    :math:`\mathcal{ALC}` description logic language. In contrast with
    :class:`ELModule <mowl.nn.el.elmodule.ELModule>`, which exposes one loss
    function per :math:`\mathcal{EL}` normal form, :math:`\mathcal{ALC}` allows
    arbitrarily nested class expressions (including negation, disjunction and
    universal restrictions). Therefore the interface is defined in terms of:

    * :meth:`forward`, which computes the loss of a (grouped) axiom, and
    * :meth:`forward_fs`, which recursively computes the *fuzzy set* membership
      degree of a class expression over a set of (anonymous) entity embeddings.

    The fuzzy logical operators (:meth:`_logical_and`, :meth:`_logical_or`,
    :meth:`_logical_not`, :meth:`_logical_exist`, :meth:`_logical_forall` and
    :meth:`_logical_residuum`) are declared here as part of the interface; their
    concrete semantics (e.g. product t-norm, Łukasiewicz t-norm) are chosen by the
    subclass. More information can be found at :doc:`/embedding_el/index`.
    """

    def forward(self, axiom, x, e_emb, *args, **kwargs):
        """Computes the loss of an axiom given an entity-membership context.

        :param axiom: The (grouped) axiom or axiom pattern to score. Typically an
            instance of :class:`org.semanticweb.owlapi.model.OWLAxiom`.
        :param x: Tensor encoding the concrete classes, properties and individuals
            that fill the axiom pattern.
        :type x: :class:`torch.Tensor`
        :param e_emb: Embeddings of the entities (including sampled anonymous
            entities) over which fuzzy memberships are evaluated.
        :type e_emb: :class:`torch.Tensor`
        """
        raise NotImplementedError()

    def forward_fs(self, cexpr, x, e_emb, cur_index=0):
        """Recursively computes the fuzzy-set membership degree of a class
        expression over the entities ``e_emb``.

        :param cexpr: Class expression of type
            :class:`org.semanticweb.owlapi.model.OWLClassExpression`.
        :param x: Tensor encoding the named classes/properties of ``cexpr``.
        :type x: :class:`torch.Tensor`
        :param e_emb: Entity embeddings.
        :type e_emb: :class:`torch.Tensor`
        :param cur_index: Current column offset into ``x`` while walking the
            expression tree. Defaults to ``0``.
        :type cur_index: int, optional
        """
        raise NotImplementedError()

    def _logical_and(self, x, y):
        """Fuzzy conjunction (t-norm) of membership degrees ``x`` and ``y``."""
        raise NotImplementedError()

    def _logical_or(self, x, y):
        """Fuzzy disjunction (t-conorm) of membership degrees ``x`` and ``y``."""
        raise NotImplementedError()

    def _logical_not(self, x):
        """Fuzzy negation of membership degrees ``x``."""
        raise NotImplementedError()

    def _logical_exist(self, r_fs, c_fs):
        """Fuzzy existential restriction :math:`\\exists R. C`."""
        raise NotImplementedError()

    def _logical_forall(self, r_fs, c_fs):
        """Fuzzy universal restriction :math:`\\forall R. C`."""
        raise NotImplementedError()

    def _logical_residuum(self, r_fs, c_fs):
        """Fuzzy residuum (implication) used by the universal restriction."""
        raise NotImplementedError()
