import torch
from torch.utils import checkpoint
from mowl.owlapi import (
    OWLAPIAdapter, ClassExpressionType, OWLSubClassOfAxiom,
    OWLEquivalentClassesAxiom)


class FALCONModule(torch.nn.Module):
    """Based on the original implementation at https://github.com/bio-ontology-research-group/FALCON
    """
    
    def __init__(
            self, nclasses, nentities, nrelations, embed_dim=128,
            anon_e=4, t_norm='product', max_measure='max', residuum='notCorD',
            device='cpu'):
        super().__init__()
        self.nentities = nentities + anon_e
        self.anon_e = anon_e

        self.c_embedding = torch.nn.Embedding(nclasses, embed_dim)
        self.r_embedding = torch.nn.Embedding(nrelations, embed_dim)
        self.e_embedding = torch.nn.Embedding(nentities, embed_dim)
        self.fc_0 = torch.nn.Linear(embed_dim * 2, 1)
        
        torch.nn.init.xavier_uniform_(self.c_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.r_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.e_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_0.weight.data)
        
        self.max_measure = max_measure
        self.t_norm = t_norm
        self.nothing = torch.zeros(self.nentities).to(device)
        self.residuum = residuum
        self.device = device
        self.adapter = OWLAPIAdapter()
    
    def _logical_and(self, x, y):
        if self.t_norm == 'product':
            return x * y
        elif self.t_norm == 'minmax':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return torch.cat([x, y], dim=-2).min(dim=-2)[0]
        elif self.t_norm == 'Łukasiewicz':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return (((x + y -1) > 0) * (x + y - 1)).squeeze(dim=-2)
        else:
            raise ValueError

    def _logical_or(self, x, y):
        if self.t_norm == 'product':
            return x + y - x * y
        elif self.t_norm == 'minmax':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return torch.cat([x, y], dim=-2).max(dim=-2)[0]
        elif self.t_norm == 'Łukasiewicz':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return 1 - ((((1-x) + (1-y) -1) > 0) * ((1-x) + (1-y) - 1)).squeeze(dim=-2)
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

    def _get_c_fs(self, c_emb, anon_e_emb):
        e_emb = torch.cat([self.e_embedding.weight, anon_e_emb], dim=0)
        c_emb = c_emb.expand_as(e_emb)
        emb = torch.cat([c_emb, e_emb], dim=-1)
        # return torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
        return torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)

    def _get_c_fs_batch(self, c_emb, anon_e_emb):
        e_emb = torch.cat([self.e_embedding.weight, anon_e_emb], dim=0).unsqueeze(dim=0).repeat(c_emb.size()[0], 1, 1)
        c_emb = c_emb.unsqueeze(dim=1).expand_as(e_emb)
        emb = torch.cat([c_emb, e_emb], dim=-1)
        # return torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
        return torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)

    def _get_r_fs(self, r_emb, anon_e_emb):
        e_emb = torch.cat([self.e_embedding.weight, anon_e_emb], dim=0)
        l_emb = (e_emb + r_emb.unsqueeze(dim=0)).unsqueeze(dim=1).repeat(1, self.nentities, 1)
        r_emb = e_emb.unsqueeze(dim=0).repeat(self.nentities, 1, 1)
        emb = torch.cat([l_emb, r_emb], dim=-1)
        # return torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
        return torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)

    def _get_r_fs_batch(self, r_emb, anon_e_emb):
        e_emb = torch.cat([self.e_embedding.weight, anon_e_emb], dim=0).unsqueeze(dim=0).repeat(r_emb.size()[0], 1, 1)
        l_emb = (e_emb + r_emb.unsqueeze(dim=1).expand_as(e_emb)).unsqueeze(dim=1).repeat(1, self.nentities, 1, 1)
        r_emb = e_emb.unsqueeze(dim=2).repeat(1, 1, self.nentities, 1)
        emb = torch.cat([l_emb, r_emb], dim=-1)
        # return torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
        return torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)

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
            return - torch.log(1 - fs.max(dim=-1)[0] + 1e-10)
        elif self.max_measure[:5] == 'pmean':
            p = int(self.max_measure[-1])
            return - torch.log(1 - ((fs ** p).mean(dim=-1))**(1/p) + 1e-10)
        else:
            raise ValueError

    def forward(self, axiom, x, anon_e_emb):
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
            pass
        elif isinstance(axiom, OWLObjectPropertyAssertionAxiom):
            pass
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
        r_fs = self._get_c_fs_batch((e_emb + r_emb).view(-1, e_emb.size()[-1]), anon_e_emb).view(e_emb.size()[0], e_emb.size()[1], -1)
        c_fs = self._get_c_fs_batch(c_emb, anon_e_emb).unsqueeze(dim=1)
        dofm = self._logical_exist(r_fs, c_fs)
        return ( - torch.log(dofm[:, 0] + 1e-10).mean() - torch.log(1 - dofm[:, 1:] + 1e-10).mean()) / 2

    def forward_abox_ec_created(self, x):
        e_emb = self.e_embedding(x[:, :, 0])
        c_emb = self.c_embedding(x[:, :, 1])
        emb = torch.cat([c_emb, e_emb], dim=-1)
        if self.cfg.loss_type == 'c':
            dofm = torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)
            # dofm = torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
            return ( - torch.log(dofm[:, 0] + 1e-10).mean() - torch.log(1 - dofm[:, 1:] + 1e-10).mean()) / 2
        elif self.cfg.loss_type == 'r':
            dofm = self.fc_0(emb).squeeze(dim=-1)
            # dofm = self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1)).squeeze(dim=-1)
            return - torch.nn.functional.logsigmoid(dofm[:, 0].unsqueeze(dim=-1) - dofm[:, 1:]).mean()
        else:
            raise ValueError

    def forward_ggi(self, x, stage='train'):
        e_1_emb = self.e_embedding(x[:, :, 0])
        r_emb = self.r_embedding(x[:, :, 1])
        e_2_emb = self.e_embedding(x[:, :, 2])
        emb = torch.cat([e_1_emb + r_emb, e_2_emb], dim=-1)
        if stage == 'train':
            if self.cfg.loss_type == 'c':
                dofm = torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)
                # dofm = torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
                return ( - torch.log(dofm[:, 0] + 1e-10).mean() - torch.log(1 - dofm[:, 1:] + 1e-10).mean()) / 2
            elif self.cfg.loss_type == 'r':
                dofm = self.fc_0(emb).squeeze(dim=-1)
                # dofm = self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1)).squeeze(dim=-1)
                return - torch.nn.functional.logsigmoid(dofm[:, 0].unsqueeze(dim=-1) - dofm[:, 1:]).mean()
            else:
                raise ValueError
        elif stage == 'test':
            return torch.sigmoid(self.fc_0(emb)).flatten()
