
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "examples/zsl/plot_1_deepgozero.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_examples_zsl_plot_1_deepgozero.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_zsl_plot_1_deepgozero.py:


DeepGOZero
===========

This example corresponds to the paper `DeepGOZero: improving protein function prediction from sequence and zero-shot learning based on ontology axioms <https://doi.org/10.1093/bioinformatics/btac256>`_. DeepGOZero is a machine learning model that performs protein function prediction for functions that have small number or zero annotations.

.. GENERATED FROM PYTHON SOURCE LINES 11-12

First, we have the necesary imports for this example.

.. GENERATED FROM PYTHON SOURCE LINES 12-40

.. code-block:: Python

    import mowl
    mowl.init_jvm("10g")
    import click as ck
    import pandas as pd
    import torch as th
    import numpy as np
    from torch import nn
    from torch.nn import functional as F
    from torch import optim
    from torch.optim.lr_scheduler import MultiStepLR
    from sklearn.metrics import roc_curve, auc
    import math
    from mowl.utils.data import FastTensorDataLoader
    import os
    import pickle as pkl
    from tqdm import tqdm

    from mowl.owlapi.defaults import BOT, TOP
    from mowl.datasets import ELDataset, RemoteDataset
    from mowl.nn import ELEmModule
    from mowl.owlapi import OWLAPIAdapter
    from mowl.datasets.base import Entities, OWLClasses, OWLIndividuals

    from org.semanticweb.owlapi.model import AxiomType
    from org.semanticweb.owlapi.model.parameters import Imports
    from org.semanticweb.owlapi.reasoner.structural import StructuralReasonerFactory









.. GENERATED FROM PYTHON SOURCE LINES 41-46

Dataset
--------

The datasets are stored in the cloud and the following links correspond for the data for the
Gene Ontology sub-ontologies: molecular function, biological process and cellular component.

.. GENERATED FROM PYTHON SOURCE LINES 46-52

.. code-block:: Python


    MF_URL = "https://deepgo.cbrc.kaust.edu.sa/data/deepgozero/mowl/molecular_function.tar.gz"
    BP_URL = "https://deepgo.cbrc.kaust.edu.sa/data/deepgozero/mowl/biological_process.tar.gz"

    CC_URL = "https://deepgo.cbrc.kaust.edu.sa/data/deepgozero/mowl/cellular_component.tar.gz"








.. GENERATED FROM PYTHON SOURCE LINES 53-61

To begin, each subontology data is encapsutaled in the ``DGZeroDataset``. This class contains \
three ontologies: training, validation and testing.
For this project, the training ontology is the Gene Ontology extended with the following axioms:

* :math:`\exists has\_function. go\_class (protein)`, which encodes protein function annotations.
* :math:`has\_interpro (protein, interpro)`, which encodes interpro features for proteins.

The validation and testing ontologies contain protein function and intepro annotations.

.. GENERATED FROM PYTHON SOURCE LINES 61-175

.. code-block:: Python


    class DGZeroDataset(RemoteDataset):
        def __init__(self, subontology):
            if subontology == "mf":
                url = MF_URL
                root = "molecular_function/"
            elif subontology == "bp":
                url = BP_URL
                root = "biological_process/"
            elif subontology == "cc":
                url = CC_URL
                root = "cellular_component/"
            else:
                raise ValueError("Invalid subontology: {}".format(subontology))

            train_owl_file = root + "mowl_train.owl"
            valid_owl_file = root + "mowl_valid.owl"
            test_owl_file = root + "mowl_test.owl"

            super().__init__(url)
        
            self._proteins = None
            self._functions = None
            self._interpros = None
        
        @property
        def functions(self):
            if self._functions is None:
                functions = set()
                for cls_str, cls_owl in self.classes.as_dict.items():
                    if cls_str.startswith("http://purl.obolibrary.org/obo/GO"):
                        functions.add(cls_owl)
                self._functions = OWLClasses(functions)
            return self._functions

        @property
        def proteins(self):
            if self._proteins is None:
                proteins = set()
                for ind_str, ind_owl in self.individuals.as_dict.items():
                    if ind_str.startswith("http://mowl/protein"):
                        proteins.add(ind_owl)
                self._proteins = OWLIndividuals(proteins)
            return self._proteins

        @property
        def interpros(self):
            if self._interpros is None:
                interpros = set()
                for ind_str, ind_owl in self.individuals.as_dict.items():
                    if ind_str.startswith("http://mowl/interpro"):
                        interpros.add(ind_owl)
                self._interpros = OWLIndividuals(interpros)
            return self._interpros
    

        @property
        def evaluation_property(self):
            return "http://mowl/has_function"



    def load_data(dataset, term_to_id, ipr_to_id):
        train_data = get_data(dataset.ontology, term_to_id, ipr_to_id)
        valid_data = get_data(dataset.validation, term_to_id, ipr_to_id)
        test_data  = get_data(dataset.testing, term_to_id, ipr_to_id)
    
        return train_data, valid_data, test_data

    def get_data(ontology, term_to_id, ipr_to_id):
        axioms = ontology.getABoxAxioms(Imports.fromBoolean(False))
    
        pf_axioms = set()
        interpro_axioms = set()
    
        for abox_axiom in axioms:
            ax_name = abox_axiom.getAxiomType()
        
            if ax_name == AxiomType.CLASS_ASSERTION:
                pf_axioms.add(abox_axiom)
            elif ax_name == AxiomType.OBJECT_PROPERTY_ASSERTION:
                interpro_axioms.add(abox_axiom)
            else:
                print(f"Ignoring axiom: {abox_axiom.toString()}")
    
        individuals = ontology.getIndividualsInSignature()
        proteins = [str(i.toStringID()) for i in individuals if str(i.toStringID()).startswith("http://mowl/protein/")]
        proteins = sorted(proteins)
        prot_to_id = {p: i for i, p in enumerate(proteins)}

        data = th.zeros((len(proteins), len(ipr_to_id)), dtype=th.float32)
        labels = th.zeros((len(proteins), len(term_to_id)), dtype=th.float32)
    
        interpro_count = 0
        function_count = 0
        for axiom in interpro_axioms:
            protein = str(axiom.getSubject().toStringID())
            interpro = str(axiom.getObject().toStringID())
        
            if interpro in ipr_to_id:
                data[prot_to_id[protein], ipr_to_id[interpro]] = 1
                interpro_count += 1

        for axiom in pf_axioms:
            protein = str(axiom.getIndividual().toStringID())
            function = str(axiom.getClassExpression().getFiller().toStringID())
        
            if function in term_to_id:
                labels[prot_to_id[protein], term_to_id[function]] = 1
                function_count += 1
    
        print(f"In get_data. Interpros processed: {interpro_count}. Functions processed: {function_count}")
        return data, labels








.. GENERATED FROM PYTHON SOURCE LINES 176-183

DeepGoZero model
----------------

The DeepGoZero model is composed by:
- A protein encoder model that takes protein interpro features and learns a latent \
representation of the protein. Futhermore, this representation is associated to a GO term \
to predict if the GO term is a function of the protein.

.. GENERATED FROM PYTHON SOURCE LINES 183-212

.. code-block:: Python


    class Residual(nn.Module):

        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x):
            return x + self.fn(x)
    
        
    class MLPBlock(nn.Module):

        def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.1, activation=nn.ReLU):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias)
            self.activation = activation()
            self.layer_norm = nn.BatchNorm1d(out_features) if layer_norm else None
            self.dropout = nn.Dropout(dropout) if dropout else None

        def forward(self, x):
            x = self.activation(self.linear(x))
            if self.layer_norm:
                x = self.layer_norm(x)
            if self.dropout:
                x = self.dropout(x)
            return x









.. GENERATED FROM PYTHON SOURCE LINES 213-216

The GO terms representations are learned using a model theoretic approach called
:doc:`ELEmbeddings </examples/elmodels/plot_1_elembeddings>`. ELEmbeddings processes the axioms
of the Gene Ontology and learns a representation of the GO terms.

.. GENERATED FROM PYTHON SOURCE LINES 216-288

.. code-block:: Python

    
    class DGELModel(nn.Module):

        def __init__(self, nb_iprs, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim=1024, embed_dim=1024, margin=0.1):
            super().__init__()
            self.nb_gos = nb_gos
            self.nb_zero_gos = nb_zero_gos
            input_length = nb_iprs
            net = []
            net.append(MLPBlock(input_length, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            self.net = nn.Sequential(*net)

            # ELEmbeddings
            self.embed_dim = embed_dim
            self.hasFuncIndex = th.LongTensor([nb_rels]).to(device)
            go_embed = nn.Embedding(nb_gos + nb_zero_gos+2, embed_dim)
            #self.go_norm = nn.BatchNorm1d(embed_dim)
            k = math.sqrt(1 / embed_dim)
            nn.init.uniform_(go_embed.weight, -k, k)
            go_rad = nn.Embedding(nb_gos + nb_zero_gos, 1)
            nn.init.uniform_(go_rad.weight, -k, k)
        
            rel_embed = nn.Embedding(nb_rels + 1, embed_dim)
            nn.init.uniform_(rel_embed.weight, -k, k)
            self.all_gos = th.arange(self.nb_gos).to(device)
            self.margin = margin

            self.elembeddings = ELEmModule(nb_gos + nb_zero_gos + 2, nb_rels+1, embed_dim=embed_dim) # +2 to add top and bottom
            self.elembeddings.class_embed = go_embed
            self.elembeddings.class_rad = go_rad
            self.elembeddings.rel_embed = rel_embed
        
     
        def forward(self, features, data = None):
            if data is None:
                data = self.all_gos

            class_embed = self.elembeddings.class_embed
            rel_embed = self.elembeddings.rel_embed
            class_rad = self.elembeddings.class_rad
            x = self.net(features)
            go_embed = class_embed(data)
            hasFunc = rel_embed(self.hasFuncIndex)
            hasFuncGO = go_embed + hasFunc
            go_rad = th.abs(class_rad(data).view(1, -1))
            x = th.matmul(x, hasFuncGO.T) + go_rad
            logits = th.sigmoid(x)
            return logits

        def predict_zero(self, features, data):
            return self.forward(features, data=data)
    
        def el_loss(self, go_normal_forms):
            gci0, gci1, gci2, gci3, _ = go_normal_forms
        
            gci0_loss = self.elembeddings(gci0, "gci0")
            gci1_loss = self.elembeddings(gci1, "gci1")
            gci2_loss = self.elembeddings(gci2, "gci2")
            gci3_loss = self.elembeddings(gci3, "gci3")
            return gci0_loss.mean() + gci1_loss.mean() + gci2_loss.mean() + gci3_loss.mean()

    

    def compute_roc(labels, preds):
        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
        roc_auc = auc(fpr, tpr)

        return roc_auc









.. GENERATED FROM PYTHON SOURCE LINES 289-301

Training DeepGoZero
-------------------

In the training phase, both the protein and GO term model are trained jointly. In the model, the
objective function is composed by two terms:
- The first term is the cross entropy loss between the predicted GO term and the true GO term
for a protein
- The second term is the ELEmbeddings loss that is computed using the axioms of the Gene Ontology

Not all the GO terms are present in the first component, but only on the second component.
However, DeepGOZero is able to predict protein functions that do not have annotations by
leveraging the semantics of the Gene Ontology.

.. GENERATED FROM PYTHON SOURCE LINES 301-536

.. code-block:: Python


    def main(ont, batch_size, epochs, device):

        if not os.path.exists(f"data/{ont}"):
            os.makedirs(f"data/{ont}")
    
        print("Loading DeepGOZero dataset...")
        dataset = DGZeroDataset(ont)
    
        model_file = f'data/{ont}/deepgozero_zero_10.th'
        terms_file = str(dataset.root) + '/terms_zero_10.pkl'
        iprs_file = str(dataset.root) + '/interpros.pkl'
        out_file = str(dataset.root) + '/predictions_deepgozero_zero_10.pkl'

        functions = dataset.functions.as_str
        function_to_id = {f: i for i,f in enumerate(functions)}

        proteins = dataset.proteins.as_str
        protein_to_id = {p: i for i, p in enumerate(proteins)}

        interpros = dataset.interpros.as_str
        interpro_to_id = {ip: i for i, ip in enumerate(interpros)}

        relations = dataset.object_properties.as_str
        relation_to_id = {r: i for i, r in enumerate(relations) if r != "http://mowl/has_function"}

        print(f"Functions:\t{len(functions)}")
        print(f"Proteins: \t{len(proteins)}")
        print(f"Interpros:\t{len(interpros)}")
        print(f"Relations:\t{len(relations)}")


        # List of GO terms to be used
        terms_df = pd.read_pickle(terms_file)
        terms = terms_df['gos'].values.flatten()
        terms = ["http://purl.obolibrary.org/obo/" + t.replace(":", "_") for t in terms]
        term_to_id = {t: i for i, t in enumerate(terms)}
        n_terms = len(terms)
    
        # List of Interpros to be used
        ipr_df = pd.read_pickle(iprs_file)
        iprs = ipr_df['interpros'].values.flatten()
        iprs = ["http://mowl/interpro/" + i for i in iprs]
        ipr_to_id = {v:k for k, v in enumerate(iprs)}
        n_interpros = len(iprs)
    
        print(f"GO terms list: {n_terms}")
        print(f"Interpro list: {n_interpros}")


        z_count = 0
        z_functions = set()
        for function in functions:
            if not function in terms:
                z_functions.add(function)
                z_count += 1

        print(f'Non-zero functions:\t{n_terms}\nZero functions: \t{z_count}')

 

        zero_functions = {t: i + len(terms) for i, t in enumerate(z_functions)}
        class_to_id = {**term_to_id,  **zero_functions}
        class_to_id[BOT] = len(class_to_id)
        class_to_id[TOP] = len(class_to_id)

        # Protein function data
        train_data, valid_data, test_data = load_data(dataset, term_to_id, ipr_to_id)

        # GO data as EL
        nfs_file = f"data/{ont}/nfs.pkl"
        if os.path.exists(nfs_file):
            print("Loading normal forms from disk...")
            with open(nfs_file, "rb") as f:
                nfs = pkl.load(f)
        else:
            print("Generating EL dataset...")
            el_dataset = ELDataset(dataset.ontology, 
                                   class_index_dict=class_to_id,
                                   object_property_index_dict=relation_to_id, 
                                   extended=False)

            nfs = el_dataset.get_gci_datasets()    
            with open(nfs_file, "wb") as f:
                pkl.dump(nfs, f)

        gci0_ds = nfs["gci0"]
        gci1_ds = nfs["gci1"]
        gci2_ds = nfs["gci2"]
        gci3_ds = nfs["gci3"]
        print(f"Axioms in GCI0: {len(gci0_ds)}")
        print(f"Axioms in GCI1: {len(gci1_ds)}")
        print(f"Axioms in GCI2: {len(gci2_ds)}")
        print(f"Axioms in GCI3: {len(gci3_ds)}")

        nfs = list(nfs.values())

        n_rels = len(relation_to_id)
        n_zeros = len(zero_functions)

        net = DGELModel(n_interpros, n_terms, n_zeros, n_rels, device).to(device)
        print(net)

        train_features, train_labels = train_data
        valid_features, valid_labels = valid_data
        test_features, test_labels = test_data

        train_loader = FastTensorDataLoader(
            *train_data, batch_size=batch_size, shuffle=True)
        valid_loader = FastTensorDataLoader(
            *valid_data, batch_size=batch_size, shuffle=False)
        test_loader = FastTensorDataLoader(
            *test_data, batch_size=batch_size, shuffle=False)

        valid_labels = valid_labels.detach().cpu().numpy()
        test_labels = test_labels.detach().cpu().numpy()

        optimizer = th.optim.Adam(net.parameters(), lr=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[5, 20], gamma=0.1)

        best_loss = 10000.0
    
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_elloss = 0
            lmbda = 0.1
            train_steps = 2 # int(math.ceil(len(train_labels) / batch_size))

            count = 0
            for batch_features, batch_labels in tqdm(train_loader, total=train_steps):
                if count == train_steps:
                    break
                count += 1
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits = net(batch_features)
                loss = F.binary_cross_entropy(logits, batch_labels)
                el_loss = net.el_loss(nfs)
                total_loss = loss + el_loss
                train_loss += loss.detach().item()
                train_elloss = el_loss.detach().item()
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            train_loss /= train_steps

            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                valid_loss = 0
                preds = []

                for batch_features, batch_labels in tqdm(valid_loader, total=valid_steps):
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = net(batch_features)
                    batch_loss = F.binary_cross_entropy(logits, batch_labels)
                    valid_loss += batch_loss.detach().item()
                    preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                roc_auc = compute_roc(valid_labels, preds)
                print(f'Epoch {epoch}: Loss - {train_loss}, EL Loss: {train_elloss}, Valid loss - {valid_loss}, AUC - {roc_auc}')

            print('EL Loss', train_elloss)
            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(net.state_dict(), model_file)

            scheduler.step()


        # Loading best model
        print('Loading the best model')
        net.load_state_dict(th.load(model_file))
        net.eval()
        with th.no_grad():
            test_steps = int(math.ceil(len(test_labels) / batch_size))
            test_loss = 0
            preds = []
        
            for batch_features, batch_labels in tqdm(test_loader, total=test_steps):
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits = net(batch_features)
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                test_loss += batch_loss.detach().cpu().item()
                preds = np.append(preds, logits.detach().cpu().numpy())
            test_loss /= test_steps
            preds = preds.reshape(-1, n_terms)
            roc_auc = compute_roc(test_labels, preds)
            print(f'Test Loss - {test_loss}, AUC - {roc_auc}')

        preds = list(preds)


        adapter = OWLAPIAdapter()
        manager = adapter.owl_manager

        # Propagate scores using ontology structure


        reasoner = StructuralReasonerFactory().createReasoner(dataset.ontology)

    

        for i, scores in tqdm(enumerate(preds[:10]), total=len(preds[:10])):
            prop_annots = {}
            sup_processed = 0
            for go_id, j in term_to_id.items():
                score = scores[j]
                go_class = adapter.create_class(go_id)
                superclasses = reasoner.getSuperClasses(go_class, False).getFlattened()
                superclasses = [str(sup.toStringID()) for sup in superclasses]
                for sup_go in superclasses:
                    if sup_go in prop_annots:
                        prop_annots[sup_go] = max(prop_annots[sup_go], score)
                        sup_processed += 1
                    else:
                        prop_annots[sup_go] = score
            for go_id, score in prop_annots.items():
                if go_id in term_to_id:
                    scores[term_to_id[go_id]] = score



        # TODO: refactor this to save predictions in an .owl file
        # test_df['preds'] = preds
        # test_df.to_pickle(out_file)









.. GENERATED FROM PYTHON SOURCE LINES 537-539

Training the model
--------------------

.. GENERATED FROM PYTHON SOURCE LINES 539-546

.. code-block:: Python



    ont = "mf"
    batch_size = 16
    epochs = 3
    device = "cpu"
    main(ont, batch_size, epochs, device)




.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    Loading DeepGOZero dataset...
    Functions:      50722
    Proteins:       43279
    Interpros:      21579
    Relations:      11
    GO terms list: 2041
    Interpro list: 26406
    Non-zero functions:     2041
    Zero functions:         48681
    In get_data. Interpros processed: 153955. Functions processed: 364571
    In get_data. Interpros processed: 17956. Functions processed: 40176
    In get_data. Interpros processed: 21084. Functions processed: 51317
    Loading normal forms from disk...
    /home/zhapacfp/miniforge3/envs/mowldev39/lib/python3.9/site-packages/torch/storage.py:414: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      return torch.load(io.BytesIO(b))
    Axioms in GCI0: 80941
    Axioms in GCI1: 11842
    Axioms in GCI2: 19594
    Axioms in GCI3: 11810
    DGELModel(
      (net): Sequential(
        (0): MLPBlock(
          (linear): Linear(in_features=26406, out_features=1024, bias=True)
          (activation): ReLU()
          (layer_norm): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (1): Residual(
          (fn): MLPBlock(
            (linear): Linear(in_features=1024, out_features=1024, bias=True)
            (activation): ReLU()
            (layer_norm): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (elembeddings): ELEmModule(
        (class_embed): Embedding(50724, 1024)
        (class_rad): Embedding(50722, 1)
        (rel_embed): Embedding(11, 1024)
      )
    )
    Training the model
      0%|          | 0/2 [00:00<?, ?it/s]     50%|█████     | 1/2 [00:01<00:01,  1.39s/it]    100%|██████████| 2/2 [00:03<00:00,  1.54s/it]    100%|██████████| 2/2 [00:03<00:00,  1.52s/it]
    Validation
      0%|          | 0/241 [00:00<?, ?it/s]      6%|▌         | 14/241 [00:00<00:01, 134.94it/s]     12%|█▏        | 28/241 [00:00<00:01, 132.28it/s]     17%|█▋        | 42/241 [00:00<00:01, 129.89it/s]     23%|██▎       | 55/241 [00:00<00:01, 126.41it/s]     28%|██▊       | 68/241 [00:00<00:01, 120.45it/s]     34%|███▎      | 81/241 [00:00<00:01, 107.48it/s]     38%|███▊      | 92/241 [00:00<00:01, 100.62it/s]     43%|████▎     | 103/241 [00:00<00:01, 95.19it/s]     47%|████▋     | 113/241 [00:01<00:01, 90.71it/s]     51%|█████     | 123/241 [00:01<00:01, 87.25it/s]     55%|█████▍    | 132/241 [00:01<00:01, 84.46it/s]     59%|█████▊    | 141/241 [00:01<00:01, 81.86it/s]     62%|██████▏   | 150/241 [00:01<00:01, 79.45it/s]     66%|██████▌   | 158/241 [00:01<00:01, 77.18it/s]     69%|██████▉   | 166/241 [00:01<00:00, 75.11it/s]     72%|███████▏  | 174/241 [00:01<00:00, 73.25it/s]     76%|███████▌  | 182/241 [00:02<00:00, 71.24it/s]     79%|███████▉  | 190/241 [00:02<00:00, 69.60it/s]     82%|████████▏ | 197/241 [00:02<00:00, 68.41it/s]     85%|████████▍ | 204/241 [00:02<00:00, 66.87it/s]     88%|████████▊ | 211/241 [00:02<00:00, 65.64it/s]     90%|█████████ | 218/241 [00:02<00:00, 64.52it/s]     93%|█████████▎| 225/241 [00:02<00:00, 63.20it/s]     96%|█████████▋| 232/241 [00:02<00:00, 62.24it/s]     99%|█████████▉| 239/241 [00:02<00:00, 61.21it/s]    100%|██████████| 241/241 [00:02<00:00, 80.98it/s]
    Epoch 0: Loss - 0.8623934686183929, EL Loss: 4.422718524932861, Valid loss - 0.6801391436845929, AUC - 0.5278044320039625
    EL Loss 4.422718524932861
    Saving model
      0%|          | 0/2 [00:00<?, ?it/s]     50%|█████     | 1/2 [00:01<00:01,  1.74s/it]    100%|██████████| 2/2 [00:02<00:00,  1.40s/it]    100%|██████████| 2/2 [00:02<00:00,  1.45s/it]
    Validation
      0%|          | 0/241 [00:00<?, ?it/s]      5%|▍         | 11/241 [00:00<00:02, 105.44it/s]     10%|▉         | 24/241 [00:00<00:01, 117.83it/s]     15%|█▌        | 37/241 [00:00<00:01, 120.08it/s]     21%|██        | 50/241 [00:00<00:01, 119.52it/s]     26%|██▌       | 62/241 [00:00<00:01, 117.67it/s]     31%|███       | 74/241 [00:00<00:01, 114.83it/s]     36%|███▌      | 86/241 [00:00<00:01, 113.34it/s]     41%|████      | 98/241 [00:00<00:01, 110.97it/s]     46%|████▌     | 110/241 [00:00<00:01, 108.92it/s]     50%|█████     | 121/241 [00:01<00:01, 106.92it/s]     55%|█████▍    | 132/241 [00:01<00:01, 101.29it/s]     59%|█████▉    | 143/241 [00:01<00:01, 91.00it/s]      63%|██████▎   | 153/241 [00:01<00:01, 84.85it/s]     67%|██████▋   | 162/241 [00:01<00:00, 79.38it/s]     71%|███████   | 171/241 [00:01<00:00, 76.00it/s]     74%|███████▍  | 179/241 [00:01<00:00, 73.21it/s]     78%|███████▊  | 187/241 [00:02<00:00, 71.15it/s]     81%|████████  | 195/241 [00:02<00:00, 68.71it/s]     84%|████████▍ | 202/241 [00:02<00:00, 67.20it/s]     87%|████████▋ | 209/241 [00:02<00:00, 65.81it/s]     90%|████████▉ | 216/241 [00:02<00:00, 64.54it/s]     93%|█████████▎| 223/241 [00:02<00:00, 62.95it/s]     95%|█████████▌| 230/241 [00:02<00:00, 61.97it/s]     98%|█████████▊| 237/241 [00:02<00:00, 61.03it/s]    100%|██████████| 241/241 [00:02<00:00, 82.94it/s]
    Epoch 1: Loss - 0.9421012699604034, EL Loss: 4.164636611938477, Valid loss - 0.6609049273724378, AUC - 0.5361168305841411
    EL Loss 4.164636611938477
    Saving model
      0%|          | 0/2 [00:00<?, ?it/s]     50%|█████     | 1/2 [00:01<00:01,  1.70s/it]    100%|██████████| 2/2 [00:02<00:00,  1.42s/it]    100%|██████████| 2/2 [00:02<00:00,  1.46s/it]
    Validation
      0%|          | 0/241 [00:00<?, ?it/s]      6%|▌         | 14/241 [00:00<00:01, 135.22it/s]     12%|█▏        | 28/241 [00:00<00:01, 132.63it/s]     17%|█▋        | 42/241 [00:00<00:01, 129.42it/s]     23%|██▎       | 55/241 [00:00<00:01, 125.69it/s]     28%|██▊       | 68/241 [00:00<00:01, 123.24it/s]     34%|███▎      | 81/241 [00:00<00:01, 119.43it/s]     39%|███▊      | 93/241 [00:00<00:01, 117.26it/s]     44%|████▎     | 105/241 [00:00<00:01, 112.53it/s]     49%|████▊     | 117/241 [00:00<00:01, 109.97it/s]     54%|█████▎    | 129/241 [00:01<00:01, 106.29it/s]     58%|█████▊    | 140/241 [00:01<00:01, 95.49it/s]      62%|██████▏   | 150/241 [00:01<00:01, 88.06it/s]     66%|██████▌   | 159/241 [00:01<00:00, 82.99it/s]     70%|██████▉   | 168/241 [00:01<00:00, 78.48it/s]     73%|███████▎  | 176/241 [00:01<00:00, 75.51it/s]     76%|███████▋  | 184/241 [00:01<00:00, 72.76it/s]     80%|███████▉  | 192/241 [00:02<00:00, 70.60it/s]     83%|████████▎ | 200/241 [00:02<00:00, 68.65it/s]     86%|████████▌ | 207/241 [00:02<00:00, 67.05it/s]     89%|████████▉ | 214/241 [00:02<00:00, 65.62it/s]     92%|█████████▏| 221/241 [00:02<00:00, 64.20it/s]     95%|█████████▍| 228/241 [00:02<00:00, 62.90it/s]     98%|█████████▊| 235/241 [00:02<00:00, 61.72it/s]    100%|██████████| 241/241 [00:02<00:00, 85.25it/s]
    Epoch 2: Loss - 0.9112534523010254, EL Loss: 3.9117393493652344, Valid loss - 0.6448946365182331, AUC - 0.5311882185872345
    EL Loss 3.9117393493652344
    Saving model
    Loading the best model
    /home/zhapacfp/Git/mowl/examples/zsl/plot_1_deepgozero.py:479: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      net.load_state_dict(th.load(model_file))
      0%|          | 0/295 [00:00<?, ?it/s]      5%|▍         | 14/295 [00:00<00:02, 130.97it/s]      9%|▉         | 28/295 [00:00<00:02, 128.49it/s]     14%|█▍        | 41/295 [00:00<00:02, 125.94it/s]     18%|█▊        | 54/295 [00:00<00:01, 124.05it/s]     23%|██▎       | 67/295 [00:00<00:01, 121.30it/s]     27%|██▋       | 80/295 [00:00<00:01, 118.65it/s]     31%|███       | 92/295 [00:00<00:01, 115.68it/s]     35%|███▌      | 104/295 [00:00<00:01, 112.08it/s]     39%|███▉      | 116/295 [00:00<00:01, 109.87it/s]     43%|████▎     | 127/295 [00:01<00:01, 106.66it/s]     47%|████▋     | 138/295 [00:01<00:01, 95.29it/s]      50%|█████     | 148/295 [00:01<00:01, 87.70it/s]     53%|█████▎    | 157/295 [00:01<00:01, 82.44it/s]     56%|█████▋    | 166/295 [00:01<00:01, 78.21it/s]     59%|█████▉    | 174/295 [00:01<00:01, 74.87it/s]     62%|██████▏   | 182/295 [00:01<00:01, 72.34it/s]     64%|██████▍   | 190/295 [00:02<00:01, 70.06it/s]     67%|██████▋   | 198/295 [00:02<00:01, 67.90it/s]     69%|██████▉   | 205/295 [00:02<00:01, 66.18it/s]     72%|███████▏  | 212/295 [00:02<00:01, 64.79it/s]     74%|███████▍  | 219/295 [00:02<00:01, 63.54it/s]     77%|███████▋  | 226/295 [00:02<00:01, 62.34it/s]     79%|███████▉  | 233/295 [00:02<00:01, 61.12it/s]     81%|████████▏ | 240/295 [00:02<00:00, 59.89it/s]     83%|████████▎ | 246/295 [00:02<00:00, 59.07it/s]     85%|████████▌ | 252/295 [00:03<00:00, 58.21it/s]     87%|████████▋ | 258/295 [00:03<00:00, 57.24it/s]     89%|████████▉ | 264/295 [00:03<00:00, 56.28it/s]     92%|█████████▏| 270/295 [00:03<00:00, 55.54it/s]     94%|█████████▎| 276/295 [00:03<00:00, 54.51it/s]     96%|█████████▌| 282/295 [00:03<00:00, 53.36it/s]     98%|█████████▊| 288/295 [00:03<00:00, 52.23it/s]    100%|█████████▉| 294/295 [00:03<00:00, 51.50it/s]    100%|██████████| 295/295 [00:03<00:00, 75.95it/s]
    Test Loss - 0.6448748683525344, AUC - 0.5255426961594861
      0%|          | 0/10 [00:00<?, ?it/s]     10%|█         | 1/10 [00:00<00:01,  7.60it/s]     20%|██        | 2/10 [00:00<00:01,  7.88it/s]     30%|███       | 3/10 [00:00<00:00,  8.00it/s]     40%|████      | 4/10 [00:00<00:00,  7.98it/s]     50%|█████     | 5/10 [00:00<00:00,  8.01it/s]     60%|██████    | 6/10 [00:00<00:00,  8.02it/s]     70%|███████   | 7/10 [00:00<00:00,  6.70it/s]     80%|████████  | 8/10 [00:01<00:00,  7.25it/s]     90%|█████████ | 9/10 [00:01<00:00,  7.71it/s]    100%|██████████| 10/10 [00:01<00:00,  8.04it/s]    100%|██████████| 10/10 [00:01<00:00,  7.75it/s]





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (2 minutes 17.980 seconds)

**Estimated memory usage:**  17315 MB


.. _sphx_glr_download_examples_zsl_plot_1_deepgozero.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_1_deepgozero.ipynb <plot_1_deepgozero.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_1_deepgozero.py <plot_1_deepgozero.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_1_deepgozero.zip <plot_1_deepgozero.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
