import torch as th
import torch.nn as nn


class ELModule(nn.Module):
    """Subclass of :class:`torch.nn.Module` for :math:`\mathcal{EL}` models. 

    This class provides \
    an interface for loss functions of the 7 possible normal forms existing in the \
    :math:`\mathcal{EL}` language. In case a negative version of one of the loss function exist, \
    it must be placed inside the original loss function and be accesed through the ``neg`` \
    parameter. More information of this can be found at :doc:`/embedding_el/index`
    """

    def __init__(self):
        super().__init__()

        self.class_embed = None
        self.rel_embed = None
        self.ind_embed = None

        self.gci_names = ["gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot", "class_assertion", "object_property_assertion"]

    def gci0_loss(self, gci, neg=False):
        """Loss function for GCI0: :math:`C \sqsubseteq D`.

        :param gci: Input tensor of shape \(\*,2\) where ``C`` classes will be at ``gci[:,0]`` \
        and ``D`` classes will be at ``gci[:,1]``. It is recommended to use the \
        :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type gci: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be \
        used. Defaults to ``False``.
        :type neg: bool, optional.
        """

        raise NotImplementedError()

    def gci1_loss(self, gci, neg=False):
        """Loss function for GCI1: :math:`C_1 \sqcap C_2 \sqsubseteq D`.

        :param gci: Input tensor of shape \(\*,3\) where ``C1`` classes will be at \
        ``gci[:,0]``, ``C2`` classes will be at ``gci[:,1]`` and ``D`` classes will be at \
        ``gci[:,2]``. It is recommended to use the :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type gci: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be \
        used. Defaults to ``False``.
        :type neg: bool, optional.
        """

        raise NotImplementedError()

    def gci2_loss(self, gci, neg=False):
        """Loss function for GCI2: :math:`C \sqsubseteq \exists R.D`.
        :math:`C \sqsubseteq \exists R. D`.
        :param gci: Input tensor of shape \(\*,3\) where ``C`` classes will be at \
        ``gci[:,0]``, ``R`` object properties will be at ``gci[:,1]`` and ``D`` classes will be \
        at ``gci[:,2]``. It is recommended to use the :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type gci: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be \
        used. Defaults to ``False``.
        :type neg: bool, optional.
        """

        raise NotImplementedError()

    def gci3_loss(self, gci, neg=False):
        """Loss function for GCI3: :math:`\exists R.C \sqsubseteq D`.

        :param gci: Input tensor of shape \(\*,3\) where ``R`` object properties will be at \
        gci[:,0], ``C`` classes will be at ``gci[:,1]``  and ``D`` classes will be at \
        ``gci[:,2]``. It is recommended to use the :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type gci: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be \
        used. Defaults to ``False``.
        :type neg: bool, optional.
        """

        raise NotImplementedError()

    def gci0_bot_loss(self, gci, neg=False):
        """Loss function for GCI0 with bottom concept: :math:`C \sqsubseteq \perp`.

        :param gci: Input tensor of shape \(\*,2\) where ``C`` classes will be at ``gci[:,0]`` \
        and ``bottom`` classes will be at ``gci[:,1]``. It is recommended to use the \
        :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type gci: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be \
        used. Defaults to ``False``.
        :type neg: bool, optional.
        """

        raise NotImplementedError()

    def gci1_bot_loss(self, gci, neg=False):
        """Loss function for GCI1 with bottom concept: :math:`C_1 \sqcap C_2 \sqsubseteq \perp`.

        :param gci: Input tensor of shape \(\*,3\) where ``C1`` classes will be at ``gci[:,0]``, \
        ``C2`` classes will be at ``gci[:,1] and`` ``bottom`` classes will be at ``gci[:,2]``. It is \
        recommended to use the :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type gci: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be \
        used. Defaults to ``False``.
        :type neg: bool, optional.
        """

        raise NotImplementedError()

    def gci3_bot_loss(self, gci, neg=False):
        """Loss function for GCI3 with bottom concept: :math:`\exists R.C \sqsubseteq \perp`.

        :param gci: Input tensor of shape \(\*,3\) where ``R`` object properties will be at \
        gci[:,0], ``C`` classes will be at ``gci[:,1]``  and ``bottom`` classes will be at \
        ``gci[:,2]``. It is recommended to use the :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type gci: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be \
        used. Defaults to ``False``.
        :type neg: bool, optional.
        """

        raise NotImplementedError()


    def class_assertion_loss(self, axiom_data, neg=False):
        """Loss function for class assertion: :math:`C(a)`.
        :param axiom_data: Input tensor of shape \(\*,2\) where ``C`` classes will be at \
        ``axiom_data[:,0]`` and ``a`` individuals will be at ``axiom_data[:,1]``. It is \
        recommended to use the :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type axiom_data: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be \
        used. Defaults to ``False``.
        :type neg: bool, optional.
        """

        raise NotImplementedError()
        
    def object_property_assertion_loss(self, axiom_data, neg=False):
        """Loss function for role assertion: :math:`R(a,b)`.
        :param axiom_data: Input tensor of shape \(\*,3\) where ``a`` object properties will be at \
        ``axiom_data[0], ``R`` object properties will be at ``axiom_data[:,1]`` and ``b`` individuals \
        will be at ``axiom_data[:,2]``. It is recommended to use the :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type axiom_data: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be \
        used. Defaults to ``False``.
        :type neg: bool, optional.
        """

        return NotImplementedError()
    
    def get_loss_function(self, gci_name):
        """
        This chooses the corresponding loss fuction given the name of the GCI.

        :param gci_name: Name of the GCI. Choices are ``gci0``, ``gci1``, ``gci2``, ``gci3``, \
        ``gci0_bot``, ``gci1_bot`` and ``gci3_bot``.
        :type gci_name: str
        """

        if gci_name not in self.gci_names:
            raise ValueError(
                f"Parameter gci_name must be one of the following: {', '.join(self.gci_names)}.")

        if gci_name == "gci2_bot":
            raise ValueError("GCI2 does not allow bottom entity in the right side.")
        return {
            "gci0_bot": self.gci0_bot_loss,
            "gci1_bot": self.gci1_bot_loss,
            "gci3_bot": self.gci3_bot_loss,
            "gci0": self.gci0_loss,
            "gci1": self.gci1_loss,
            "gci2": self.gci2_loss,
            "gci3": self.gci3_loss,
            "class_assertion": self.class_assertion_loss,
            "object_property_assertion": self.object_property_assertion_loss
        }[gci_name]

    def forward(self, gci, gci_name, neg=False):
        loss_fn = self.get_loss_function(gci_name)

        loss = loss_fn(gci, neg=neg)
        return loss
