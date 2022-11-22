import torch as th
from torch.utils import checkpoint
from mowl.owlapi import OWLAPIAdapter, ClassExpressionType, OWLSubClassOfAxiom, \
    OWLEquivalentClassesAxiom, OWLDisjointClassesAxiom, OWLClassAssertionAxiom, \
    OWLObjectPropertyAssertionAxiom


class FALCONModule(th.nn.Module):
    """Based on the original implementation at \
https://github.com/bio-ontology-research-group/FALCON
    """

    def __init__(
            self, nclasses, nentities, nrelations, heads_dict, tails_dict, embed_dim=128,
            anon_e=4, t_norm='product', max_measure='max', residuum='notCorD',
            loss_type='c', num_negs=4, device='cpu'):
        super().__init__()
        self.nentities = nentities
        self.anon_e = anon_e
        self.heads_dict = heads_dict
        self.tails_dict = tails_dict
        self.c_embedding = th.nn.Embedding(nclasses, embed_dim)
        self.r_embedding = th.nn.Embedding(nrelations, embed_dim)
        self.e_embedding = th.nn.Embedding(nentities, embed_dim)
        self.fc_0 = th.nn.Linear(embed_dim * 2, 1)

        th.nn.init.xavier_uniform_(self.c_embedding.weight.data)
        th.nn.init.xavier_uniform_(self.r_embedding.weight.data)
        th.nn.init.xavier_uniform_(self.e_embedding.weight.data)
        th.nn.init.xavier_uniform_(self.fc_0.weight.data)

        self.max_measure = max_measure
        self.t_norm = t_norm
        self.loss_type = loss_type
        self.num_negs = num_negs
        self.nothing = th.zeros(self.nentities).to(device)
        self.zero_emb = th.zeros(embed_dim).to(device)
        self.residuum = residuum
        self.device = device
        self.adapter = OWLAPIAdapter()

    def _logical_and(self, x, y):
        if self.t_norm == 'product':
            return x * y
        elif self.t_norm == 'minmax':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return th.cat([x, y], dim=-2).min(dim=-2)[0]
        elif self.t_norm == 'Łukasiewicz':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return (((x + y - 1) > 0) * (x + y - 1)).squeeze(dim=-2)
        else:
            raise ValueError

    def _logical_or(self, x, y):
        if self.t_norm == 'product':
            return x + y - x * y
        elif self.t_norm == 'minmax':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return th.cat([x, y], dim=-2).max(dim=-2)[0]
        elif self.t_norm == 'Łukasiewicz':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return 1 - ((((1 - x) + (1 - y) - 1) > 0) * ((1 - x) + (1 - y) - 1)).squeeze(dim=-2)
        else:
            raise ValueError

    def _logical_not(self, x):
        return 1 - x

    def _logical_residuum(self, r_fs, c_fs):
        if self.residuum == 'notCorD':
            return self._logical_or(self._logical_not(r_fs), c_fs.unsqueeze(dim=-2))
        else:
            raise ValueError

    def _logical_exist(self, r_fs, c_fs):
        return self._logical_and(r_fs, c_fs).max(dim=-1)[0]

    def _logical_forall(self, r_fs, c_fs):
        return self._logical_residuum(r_fs, c_fs).min(dim=-1)[0]

    def _get_c_fs_batch(self, c_emb, e_emb):
        e_emb = e_emb.unsqueeze(
            dim=0).repeat(c_emb.size()[0], 1, 1)
        c_emb = c_emb.unsqueeze(dim=1).expand_as(e_emb)
        emb = th.cat([c_emb, e_emb], dim=-1)
        # return th.sigmoid(self.fc_1(th.nn.functional.leaky_relu(self.fc_0(emb),
        # negative_slope=0.1))).squeeze(dim=-1)
        return th.sigmoid(self.fc_0(emb)).squeeze(dim=-1)

    def _get_r_fs_batch(self, r_emb, e_emb):
        nentities = e_emb.shape[0]
        e_emb = e_emb.unsqueeze(
            dim=0).repeat(r_emb.size()[0], 1, 1)
        l_emb = (e_emb + r_emb.unsqueeze(dim=1).expand_as(e_emb)
                 ).unsqueeze(dim=1).repeat(1, nentities, 1, 1)
        r_emb = e_emb.unsqueeze(dim=2).repeat(1, 1, nentities, 1)
        emb = th.cat([l_emb, r_emb], dim=-1)
        # return th.sigmoid(self.fc_1(th.nn.functional.leaky_relu(self.fc_0(emb),
        # negative_slope=0.1))).squeeze(dim=-1)
        return th.sigmoid(self.fc_0(emb)).squeeze(dim=-1)

    def sample_negatives(self, e, r, used_dict):
        ret = th.zeros((e.shape[0], self.num_negs), dtype=th.int64)
        for i in range(e.shape[0]):
            used = th.tensor(used_dict[(e[i].item(), r[i].item())])
            neg_pool = th.ones(self.nentities)
            neg_pool[used] = 0
            neg_pool = neg_pool.nonzero()
            neg = neg_pool[th.randint(len(neg_pool), (self.num_negs,))]
            ret[i, :] = neg.flatten()
        return ret

    def forward_fs(self, cexpr, x, anon_e_emb, cur_index=0):
        expr_type = cexpr.getClassExpressionType()
        if expr_type == ClassExpressionType.OWL_CLASS:
            c_emb = self.c_embedding(x[:, cur_index])
            return self._get_c_fs_batch(c_emb, anon_e_emb)
        elif expr_type == ClassExpressionType.OBJECT_SOME_VALUES_FROM:
            r_emb = self.r_embedding(x[:, cur_index])
            # r_fs = self._get_r_fs(r_emb, anon_e_emb)
            r_fs = checkpoint.checkpoint(self._get_r_fs_batch, r_emb, anon_e_emb)
            return self._logical_exist(
                r_fs, self.forward_fs(
                    cexpr.getFiller(), x, anon_e_emb, cur_index=cur_index + 1))
        elif expr_type == ClassExpressionType.OBJECT_ALL_VALUES_FROM:
            r_emb = self.r_embedding(x[:, cur_index])
            # r_fs = self._get_r_fs(r_emb, anon_e_emb)
            r_fs = checkpoint.checkpoint(self._get_r_fs_batch, r_emb, anon_e_emb)
            return self._logical_forall(
                r_fs, self.forward_fs(
                    cexpr.getFiller(), x, anon_e_emb, cur_index=cur_index + 1))
        elif expr_type == ClassExpressionType.OBJECT_INTERSECTION_OF:
            cexprs = [self.forward_fs(expr, x, anon_e_emb, cur_index=cur_index + i)
                      for i, expr in enumerate(cexpr.getOperandsAsList())]
            ret = cexprs[0]
            for i in range(1, len(cexprs)):
                ret = self._logical_and(ret, cexprs[i])
            return ret
        elif expr_type == ClassExpressionType.OBJECT_UNION_OF:
            cexprs = [self.forward_fs(expr, x, anon_e_emb, cur_index=cur_index + i)
                      for i, expr in enumerate(cexpr.getOperandsAsList())]
            ret = cexprs[0]
            for i in range(1, len(cexprs)):
                ret = self._logical_or(ret, cexprs[i])
            return ret
        elif expr_type == ClassExpressionType.OBJECT_COMPLEMENT_OF:
            return self._logical_not(self.forward_fs(
                cexpr.getOperand(), x, anon_e_emb, cur_index=cur_index))
        raise NotImplementedError()

    def get_cc_loss(self, fs):
        if self.max_measure == 'max':
            return - th.log(1 - fs.max(dim=-1)[0] + 1e-10)
        elif self.max_measure[:5] == 'pmean':
            p = int(self.max_measure[-1])
            return - th.log(1 - ((fs ** p).mean(dim=-1))**(1 / p) + 1e-10)
        else:
            raise ValueError

    def forward(self, axiom, x, anon_e_emb, stage='train'):
        if isinstance(axiom, OWLSubClassOfAxiom):
            C = axiom.getSubClass(),
            D = axiom.getSuperClass()
            cexpr = self.adapter.create_object_intersection_of(
                C[0], self.adapter.create_complement_of(D))
            fs = self.forward_fs(cexpr, x, anon_e_emb)
            return self.get_cc_loss(fs).mean()
        elif isinstance(axiom, OWLEquivalentClassesAxiom):
            cexprs = axiom.getClassExpressionsAsList()
            C, D = cexprs[0], cexprs[1]
            cexpr1 = self.adapter.create_object_intersection_of(
                C, self.adapter.create_complement_of(D))
            fs1 = self.forward_fs(cexpr1, x, anon_e_emb)
            cexpr2 = self.adapter.create_object_intersection_of(
                self.adapter.create_complement_of(C), D)
            fs2 = self.forward_fs(cexpr2, x, anon_e_emb)
            return self.get_cc_loss(fs1).mean() + self.get_cc_loss(fs2).mean()
        elif isinstance(axiom, OWLDisjointClassesAxiom):
            cexprs = axiom.getClassExpressionsAsList()
            C, D = cexprs[0], cexprs[1]
            cexpr = self.adapter.create_object_intersection_of(C, D)
            fs = self.forward_fs(cexpr, x, anon_e_emb)
            return self.get_cc_loss(fs).mean()
        elif isinstance(axiom, OWLClassAssertionAxiom):
            x = x.unsqueeze(dim=1)
            size = [1] * len(x.size())
            size[1] = self.num_negs
            neg_x = x.repeat(size)
            neg_ents = th.randint(self.nentities, (x.shape[0], self.num_negs))
            neg_x[:, :, 0] = neg_ents
            x = th.cat([x, neg_x], dim=1)
            cexpr = axiom.getClassExpression()
            r = None
            if cexpr.getClassExpressionType() == ClassExpressionType.OBJECT_SOME_VALUES_FROM:
                r = cexpr.getProperty()
                rx = x[:, :, 1]
                cexpr = cexpr.getFiller()
                cx = x[:, 0, 2:]
            else:
                cx = x[:, 0, 1:]
            c_fs = self.forward_fs(cexpr, cx, anon_e_emb)
            if r is not None:
                r_emb = self.r_embedding(rx)
            else:
                r_emb = 0
            ex = x[:, :, 0]
            e_emb = self.e_embedding(ex)
            r_fs = self._get_c_fs_batch(
                (e_emb + r_emb).view(-1, e_emb.shape[-1]), anon_e_emb).view(e_emb.shape[0],
                                                                            e_emb.shape[1],
                                                                            -1)
            c_fs = c_fs.unsqueeze(dim=1)
            dofm = self._logical_exist(r_fs, c_fs)
            res = (- th.log(dofm[:, 0] + 1e-10).mean() - th.log(1 - dofm[:, 1:] + 1e-10).mean())
            return res / 2
        elif isinstance(axiom, OWLObjectPropertyAssertionAxiom):
            x = x.unsqueeze(dim=1)
            size = [1] * len(x.size())
            size[1] = self.num_negs
            neg_h = x.repeat(size)
            neg_ents = self.sample_negatives(x[:, :, 2], x[:, :, 1], self.heads_dict)
            neg_h[:, :, 0] = neg_ents
            neg_t = x.repeat(size)
            neg_ents = self.sample_negatives(x[:, :, 0], x[:, :, 1], self.tails_dict)
            neg_t[:, :, 2] = neg_ents
            x = th.cat([x, neg_h, neg_t], dim=1)
            e_1_emb = self.e_embedding(x[:, :, 0])
            r_emb = self.r_embedding(x[:, :, 1])
            e_2_emb = self.e_embedding(x[:, :, 2])
            emb = th.cat([e_1_emb + r_emb, e_2_emb], dim=-1)
            if stage == 'train':
                if self.loss_type == 'c':
                    dofm = th.sigmoid(self.fc_0(emb))
                    # dofm = th.sigmoid(self.fc_1(th.nn.functional.leaky_relu(self.fc_0(emb),
                    # negative_slope=0.1))).squeeze(dim=-1)
                    res = - th.log(dofm[:, 0] + 1e-10).mean() - \
                        th.log(1 - dofm[:, 1:] + 1e-10).mean()
                    return res / 2
                elif self.loss_type == 'r':
                    dofm = self.fc_0(emb).squeeze(dim=-1)
                    # dofm = self.fc_1(th.nn.functional.leaky_relu(self.fc_0(emb),
                    # negative_slope=0.1)).squeeze(dim=-1)
                    diff = dofm[:, 0].unsqueeze(dim=-1) - dofm[:, 1:]
                    return - th.nn.functional.logsigmoid(diff).mean()
                else:
                    raise NotImplementedError()
            elif stage == 'test':
                return th.sigmoid(self.fc_0(emb)).flatten()

        else:
            raise NotImplementedError()

    def forward_name(self, x, anon_e_emb):
        c_emb_left = self.c_embedding(x[:, 0])
        c_emb_right = self.c_embedding(x[:, 2])
        fs_left = self._get_c_fs_batch(c_emb_left, anon_e_emb)
        fs_right = self._get_c_fs_batch(c_emb_right, anon_e_emb)
        return self.get_cc_loss(self._logical_and(fs_left, self._logical_not(fs_right))).mean()

    def forward_abox_ec(self, x, anon_e_emb):
        e_emb = self.e_embedding(x[:, :, 0])
        r_emb = self.r_embedding(x[0, 0, 1])
        c_emb = self.c_embedding(x[:, 0, 2])
        r_fs = self._get_c_fs_batch((e_emb + r_emb).view(-1, e_emb.size()
                                    [-1]), anon_e_emb).view(e_emb.size()[0], e_emb.size()[1], -1)
        c_fs = self._get_c_fs_batch(c_emb, anon_e_emb).unsqueeze(dim=1)
        dofm = self._logical_exist(r_fs, c_fs)
        return (- th.log(dofm[:, 0] + 1e-10).mean() - th.log(1 - dofm[:, 1:] + 1e-10).mean()) / 2

    def forward_abox_ec_created(self, x):
        e_emb = self.e_embedding(x[:, :, 0])
        c_emb = self.c_embedding(x[:, :, 1])
        emb = th.cat([c_emb, e_emb], dim=-1)
        if self.cfg.loss_type == 'c':
            dofm = th.sigmoid(self.fc_0(emb)).squeeze(dim=-1)
            # dofm = th.sigmoid(self.fc_1(th.nn.functional.leaky_relu(self.fc_0(emb),
            # negative_slope=0.1))).squeeze(dim=-1)
            res = (- th.log(dofm[:, 0] + 1e-10).mean() - th.log(1 - dofm[:, 1:] + 1e-10).mean())
            return res / 2
        elif self.cfg.loss_type == 'r':
            dofm = self.fc_0(emb).squeeze(dim=-1)
            # dofm = self.fc_1(th.nn.functional.leaky_relu(self.fc_0(emb),
            # negative_slope=0.1)).squeeze(dim=-1)
            return - th.nn.functional.logsigmoid(dofm[:, 0].unsqueeze(dim=-1) - dofm[:, 1:]).mean()
        else:
            raise ValueError

    def forward_ggi(self, x, stage='train'):
        e_1_emb = self.e_embedding(x[:, :, 0])
        r_emb = self.r_embedding(x[:, :, 1])
        e_2_emb = self.e_embedding(x[:, :, 2])
        emb = th.cat([e_1_emb + r_emb, e_2_emb], dim=-1)
        if stage == 'train':
            if self.cfg.loss_type == 'c':
                dofm = th.sigmoid(self.fc_0(emb)).squeeze(dim=-1)
                # dofm = th.sigmoid(self.fc_1(th.nn.functional.leaky_relu(self.fc_0(emb),
                # negative_slope=0.1))).squeeze(dim=-1)
                res = - th.log(dofm[:, 0] + 1e-10).mean() - th.log(1 - dofm[:, 1:] + 1e-10).mean()
                return res / 2
            elif self.cfg.loss_type == 'r':
                dofm = self.fc_0(emb).squeeze(dim=-1)
                # dofm = self.fc_1(th.nn.functional.leaky_relu(self.fc_0(emb),
                # negative_slope=0.1)).squeeze(dim=-1)
                diff = dofm[:, 0].unsqueeze(dim=-1) - dofm[:, 1:]
                return - th.nn.functional.logsigmoid(diff).mean()
            else:
                raise ValueError
        elif stage == 'test':
            return th.sigmoid(self.fc_0(emb)).flatten()
