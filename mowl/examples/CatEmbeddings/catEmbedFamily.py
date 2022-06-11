#!/usr/bin/env python
import click as ck
import numpy as np
import pandas as pd
import pickle
import gzip
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)   
sys.path.insert(0, '')
sys.path.append('../../../')

import mowl
mowl.init_jvm("10g")

import matplotlib.pyplot as plt
from mowl.datasets.base import PathDataset
from mowl.datasets.ppi_yeast import PPIYeastSlimDataset, PPIYeastDataset
from mowl.develop.catEmbeddings.modelFamily import CatEmbeddings
from mowl.visualization.base import TSNE as MTSNE
import torch as th

@ck.command()
@ck.option(
    '--species', '-sp', help="Nome of species",
    default = "yeast"
)

def main(species):
    ds = PathDataset("data/family/familyDomain.owl", None, None)
    lr = 2e-1
    embedding_size = 2
    
    gamma = 0.6
    margin = 2
    epochs = 1000
    step = 30
    milestones = [i*step for i in range(epochs//step)]
    
    device = "cuda:1"
    model = CatEmbeddings(
        ds, 
        16, #4096*4, #bs 
        embedding_size, #embeddings size
        lr, #lr ##1e-3 yeast, 1e-5 human
        epochs, #epochs
        1000, #num points eval ppi
        milestones,
        dropout = 0.2,
        decay = 0,
        gamma = gamma,
        eval_ppi = True,
        size_hom_set =2,
        depth = 1,
        margin = margin,
        seed = 0,
        early_stopping = 20000,
        device = device
    )

    model.train()

    
    classes = ["Male", "Female", "Father", "Mother", "Parent", "Person"]
    classes = ["http://" + c for c in classes]

    classes.append("http://www.w3.org/2002/07/owl#Thing")
    classes.append("http://www.w3.org/2002/07/owl#Nothing")

    #labels = {v: k for k,v in enumerate(classes)}
    
    embeddings, _ = model.get_embeddings()
    
    prod_generator = model.model.prod_net

    female = th.tensor(embeddings["http://Female"]).to("cpu")
    male = th.tensor(embeddings["http://Male"]).to("cpu")
    parent = th.tensor(embeddings["http://Parent"]).to("cpu")

    female = female.unsqueeze(0)
    male = male.unsqueeze(0)
    parent = parent.unsqueeze(0)
#    female = female.unsqueeze(0)
#    male = male.unsqueeze(0)
#    parent = parent.unsqueeze(0)
    
    female_and_parent, _ = prod_generator(female, parent)
    male_and_parent, _ = prod_generator(male, parent)

    female_and_parent = female_and_parent.squeeze()
    male_and_parent = male_and_parent.squeeze()
    female_and_parent = female_and_parent.cpu().detach().numpy()#.item()
    male_and_parent = male_and_parent.cpu().detach().numpy()#.item()

    classes.append("f_and_p")
    classes.append("m_and_p")
    labels = {c: c for c in classes}

    
    embeddings["f_and_p"] = female_and_parent
    embeddings["m_and_p"] = male_and_parent
#    embeddings = {k: v/v[-1] for k,v in embeddings.items() }
#    embeddings = {k: v - embeddings["http://www.w3.org/2002/07/owl#Nothing"] for k, v in embeddings.items()}
    print(embeddings)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    class_color_dict = {cl: col for cl, col in zip(classes, colors)}
    
    fig, ax = plt.subplots(figsize=(20, 20))


    # print(labels)
    # tsne = MTSNE(embeddings, labels, entities = classes)
    # tsne.generate_points(5000, workers = 16, verbose = 1)
    # tsne.savefig(f'data/family/tsne.jpg')
    

    
    for label, (xs,ys) in embeddings.items():
                                        
        color = class_color_dict[label]
        ax.scatter(xs, ys, color=color, label=label)

        ax.legend()
        ax.grid(True)
        
    plt.savefig("data/family/tsne.jpg")
    plt.close()
    

    

if __name__ == '__main__':
    main()
