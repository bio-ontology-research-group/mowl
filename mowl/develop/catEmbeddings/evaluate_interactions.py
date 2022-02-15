import click as ck
import numpy as np
from scipy.stats import rankdata
import torch as th
from torch.utils.data import DataLoader

def evalNF4Loss(model, test_nf4, prot_dict, prot_index, trlabels, num_prots, device = 'cpu'):

    model = model.to(device)
    top1 = 0
    top10 = 0
    top100 = 0
    mean_rank = 0
    ftop1 = 0
    ftop10 = 0
    ftop100 = 0
    fmean_rank = 0
    labels = {}
    preds = {}
    ranks = {}
    franks = {}
    eval_data = test_nf4

    cs = {c for c,r,d in eval_data}
    print("Lengths: ", len(eval_data), len(cs))
            
    n = len(eval_data)

    with ck.progressbar(eval_data) as prog_data:
        for c, r, d in prog_data:
            c, r, d = c.detach().item(), r.detach().item(), d.detach().item()
            c, d = prot_dict[c], prot_dict[d]

            if r not in labels:
                labels[r] = np.zeros((len(prot_dict), len(prot_dict)), dtype=np.int32)
            if r not in preds:
                preds[r] = np.zeros((len(prot_dict), len(prot_dict)), dtype=np.float32)
            
            labels[r][c, d] = 1

            data = th.tensor([[c,r,x] for x in prot_index]).to(device)
            
            batched_data = DataLoader(data, batch_size = 5000)
            res = []
            for i, batch in enumerate(batched_data):
                batch_res = model.nf4_loss(batch).cpu().detach().numpy()
                res = np.append(res, batch_res)
            print(res.shape, len(prot_index))
            
            # res = np.reshape((np.linalg.norm(
            #     np.maximum(euc - rd + rr , np.zeros(euc.shape)), axis=1)),
            #                  -1)  # + rightLessLeftLoss

            preds[r][c, :] = res
            index = rankdata(res, method='average')

            rank = index[d]
            
            first = np.where(index == 1)[0]
            last = np.where(index == len(prot_index))[0]

#            print(f" ({rank},{res[d]}), ({index[first]}, {res[first]}), ({index[last]}, {res[last]}) ")

            # print(1 / 0)
            
            if rank == 1:
                top1 += 1
            if rank <= 10:
                top10 += 1
            if rank <= 100:
                top100 += 1
            mean_rank += rank
            if rank not in ranks:
                ranks[rank] = 0
            ranks[rank] += 1

            # Filtered rank
           # print(res.shape,trlabels[r][c, :].shape)
            rank1 = rank
           # print(rank,1)
            index = rankdata((res * trlabels[r][c, :]), method='average')

            rank = index[d]
            
                
            #            print("\n", rank,res[d])

            
            if rank == 1:
                ftop1 += 1
            if rank <= 10:
                ftop10 += 1
            if rank <= 100:
                ftop100 += 1
            fmean_rank += rank


            if rank not in franks:
                franks[rank] = 0
            franks[rank] += 1

        top1 /= n
        top10 /= n
        top100 /= n
        mean_rank /= n
        ftop1 /= n
        ftop10 /= n
        ftop100 /= n
        fmean_rank /= n
        
    rank_auc = compute_rank_roc(ranks, num_prots)
    frank_auc = compute_rank_roc(franks, num_prots)

    return top1, top10, top100, mean_rank, ftop1, ftop10, ftop100, fmean_rank
    #print(f'{org} {embedding_size} {top10:.2f} {top100:.2f} {mean_rank:.2f} {rank_auc:.2f}')
    #print(f'{org} {embedding_size} {ftop10:.2f} {ftop100:.2f} {fmean_rank:.2f} {frank_auc:.2f}')


def compute_rank_roc(ranks, n_prots):
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n_prots)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / n_prots
    return auc
