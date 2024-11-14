from mowl.utils.data import FastTensorDataLoader
import torch as th


def create_graph_train_dataloader(graph, node_to_id, relation_to_id, batch_size):

    heads = [node_to_id[edge.src] for edge in graph]
    rels = [relation_to_id[edge.rel] for edge in graph]
    tails = [node_to_id[edge.dst] for edge in graph]

    heads = th.LongTensor(heads)
    rels = th.LongTensor(rels)
    tails = th.LongTensor(tails)

    dataloader = FastTensorDataLoader(heads, rels, tails,
                                      batch_size=batch_size, shuffle=True)
    return dataloader
