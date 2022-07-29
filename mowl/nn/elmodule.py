import torch as th
import torch.nn as nn


class ELModule(nn.Module):
    """Subclass of :class:`torch.nn.Module` for :math:`\mathcal{EL}` models. This class provides an interface for loss functions of the 7 possible normal forms existing in the :math:`\mathcal{EL}` language. In case a negative version of one of the loss function exist, it must be placed inside the original loss function and be accesed through the ``neg`` parameter. More information of this can be found at :doc:`/embedding_el/index`
    """

    def __init__(self):
        super().__init__()
        
    def gci0_loss(self, gci, neg = False):
        """Loss function for GCI0: :math:`C \sqsubseteq D`.

        :param gci: Input tensor of shape \(\*,2\) where ``C`` classes will be at ``gci[:,0]`` and ``D`` classes will be at ``gci[:,1]``. It is recommended to use the :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type gci: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be used. Defaults to ``False``.
        :type neg: bool, optional.
        """
        

       
        raise NotImplementedError()

    def gci1_loss(self, gci, neg = False):
        """Loss function for GCI1: :math:`C_1 \sqcap C_2 \sqsubseteq D`. 

        :param gci: Input tensor of shape \(\*,3\) where ``C1`` classes will be at ``gci[:,0]``, ``C2`` classes will be at gci[:,1] and ``D`` classes will be at ``gci[:,2]``. It is recommended to use the :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type gci: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be used. Defaults to ``False``.
        :type neg: bool, optional.
        """

        raise NotImplementedError()

    def gci2_loss(self, gci, neg = False):
        """Loss function for GCI2: :math:`C \sqsubseteq R. D`. 

        :param gci: Input tensor of shape \(\*,3\) where ``C`` classes will be at ``gci[:,0]``, ``R`` object properties will be at ``gci[:,1]`` and ``D`` classes will be at ``gci[:,2]``. It is recommended to use the :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type gci: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be used. Defaults to ``False``.
        :type neg: bool, optional.
        """

        raise NotImplementedError()

    def gci3_loss(self, gci, neg = False):
        """Loss function for GCI3: :math:`R. C \sqsubseteq D`. 

        :param gci: Input tensor of shape \(\*,3\) where ``R`` object properties will be at gci[:,0], ``C`` classes will be at ``gci[:,1]``  and ``D`` classes will be at ``gci[:,2]``. It is recommended to use the :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type gci: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be used. Defaults to ``False``.
        :type neg: bool, optional.
        """

        raise NotImplementedError()

    def gci0_bot_loss(self, gci, neg = False):
        """Loss function for GCI0 with bottom concept: :math:`C \sqsubseteq \perp`. 

        :param gci: Input tensor of shape \(\*,2\) where ``C`` classes will be at ``gci[:,0]`` and ``bottom`` classes will be at ``gci[:,1]``. It is recommended to use the :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type gci: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be used. Defaults to ``False``.
        :type neg: bool, optional.
        """

        raise NotImplementedError()

    def gci1_bot_loss(self, gci, neg = False):
        """Loss function for GCI1 with bottom concept: :math:`C_1 \sqcap C_2 \sqsubseteq \perp`. 

        :param gci: Input tensor of shape \(\*,3\) where ``C1`` classes will be at ``gci[:,0]``, ``C2`` classes will be at gci[:,1] and ``bottom`` classes will be at ``gci[:,2]``. It is recommended to use the :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type gci: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be used. Defaults to ``False``.
        :type neg: bool, optional.
        """

        raise NotImplementedError()

    def gci3_bot_loss(self, gci, neg = False):
        """Loss function for GCI3 with bottom concept: :math:`R. C \sqsubseteq \perp`. 

        :param gci: Input tensor of shape \(\*,3\) where ``R`` object properties will be at gci[:,0], ``C`` classes will be at ``gci[:,1]``  and ``bottom`` classes will be at ``gci[:,2]``. It is recommended to use the :class:`ELDataset <mowl.datasets.el.ELDataset>`.
        :type gci: :class:`torch.Tensor`
        :param neg: Parameter indicating that the negative version of this loss function must be used. Defaults to ``False``.
        :type neg: bool, optional.
        """

        raise NotImplementedError()

    def get_loss_function(self, gci_name):
        if gci_name == "gci2_bot":
            raise ValueError("GCI2 does not allow bottom entity in the right side.")
        return {
            "gci0_bot": self.gci0_bot_loss,
            "gci1_bot": self.gci1_bot_loss,
            "gci3_bot": self.gci3_bot_loss,
            "gci0"    : self.gci0_loss,
            "gci1"    : self.gci1_loss,
            "gci2"    : self.gci2_loss,
            "gci3"    : self.gci3_loss
        }[gci_name]

    def forward(self, gci, gci_name, neg = False):
        loss_fn = self.get_loss_function(gci_name)
        
        loss = loss_fn(gci, neg = neg)
        return loss
