"""Costs of class-assertion rows for models whose vocabulary is not the dataset's.

A model that solves or learns over its own knowledge base (its own list of concepts and
individuals) is asked for the cost of rows indexed by the *dataset's* class and individual
dictionaries. :func:`class_assertion_costs` translates the rows through the names, asks the
model's per-concept loss for each concept once, and returns an infinite cost for every row
whose class or individual is outside the model's signature (the row cannot be judged).
"""
import numpy as np
import torch as th

__all__ = ["class_assertion_costs"]


def class_assertion_costs(rows, class_index_dict, individual_index_dict, concepts,
                          individuals, concept_loss):
    """Cost of every ``[individual, class]`` row, lower is more plausible.

    :param rows: Tensor of rows of indices into ``individual_index_dict`` and
        ``class_index_dict``.
    :param concepts: Names of the concepts of the model (position = the model's index).
    :param individuals: Names of the individuals of the model.
    :param concept_loss: ``concept_loss(concept_position, [individual_positions])`` returns
        one cost per individual.
    :return: Float tensor, one cost per row, ``inf`` outside the model's signature.
    """
    concept_of = {i: iri for iri, i in class_index_dict.items()}
    individual_of = {i: iri for iri, i in individual_index_dict.items()}
    concept_pos = {iri: k for k, iri in enumerate(concepts)}
    individual_pos = {iri: k for k, iri in enumerate(individuals)}
    cpu = rows.cpu().numpy()
    out = np.full(len(cpu), np.inf)
    for c in np.unique(cpu[:, 1]):
        basic = concept_pos.get(concept_of.get(int(c)))
        if basic is None:
            continue
        where = np.flatnonzero(cpu[:, 1] == c)
        known = [(w, individual_pos.get(individual_of.get(int(cpu[w, 0])))) for w in where]
        known = [(w, a) for w, a in known if a is not None]
        if known:
            index = np.array([w for w, _ in known])
            out[index] = concept_loss(basic, [a for _, a in known])
    return th.as_tensor(out, dtype=th.float32, device=rows.device)
