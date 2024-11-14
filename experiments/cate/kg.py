import torch as th
from pykeen.models import TransE


class OrderE(TransE):
    def __init__(self, *args, **kwargs):
        super(OrderE, self).__init__(*args, **kwargs)

    def forward(self, h_indices, r_indices, t_indices, mode=None):
        h, _, t = self._get_representations(h=h_indices, r=r_indices, t=t_indices, mode=mode)
        order_loss = th.linalg.norm(th.relu(t - h), dim=1)
        return -order_loss

    def score_hrt(self, hrt_batch, mode=None):
        h, r, t = self._get_representations(h=hrt_batch[:, 0], r=hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode)
        return -th.linalg.norm(th.relu(t - h), dim=1)

    def distance(self, hrt_batch, mode=None):
        h, r, t = self._get_representations(h=hrt_batch[:, 0], r=hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode)
        mask = ((t - h) > 0).all(dim=1)
        distance = th.linalg.norm(t - h, dim=1)
        distance[mask] = -10000
        return -distance
