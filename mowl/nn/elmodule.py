import torch as th
import torch.nn as nn


class ELModule(nn.Module):

    def __init__(self, device = "cpu"):
        super().__init__()
        self.device = device

    def gci0_loss(self, gci, neg = False):
        raise NotImplementedError()

    def gci1_loss(self, gci, neg = False):
        raise NotImplementedError()

    def gci2_loss(self, gci, neg = False):
        raise NotImplementedError()

    def gci3_loss(self, gci, neg = False):
        raise NotImplementedError()

    def gci0_bot_loss(self, gci, neg = False):
        raise NotImplementedError()

    def gci1_bot_loss(self, gci, neg = False):
        raise NotImplementedError()

    def gci3_bot_loss(self, gci, neg = False):
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
