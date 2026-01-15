from mowl.ontology.normalize import ELNormalizer
from mowl.base_models.model import Model
from mowl.datasets.el import ELDataset
from mowl.projection import projector_factory
import torch as th
from torch.utils.data import DataLoader, default_collate
from tqdm import trange

from deprecated.sphinx import versionadded, versionchanged

from org.semanticweb.owlapi.model import OWLClassExpression, OWLClass, OWLObjectSomeValuesFrom, OWLObjectIntersectionOf

import copy
import numpy as np
import mowl.error.messages as msg
import os
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

@versionchanged(version="1.0.0", reason="Added the 'load_normalized' parameter.")
class EmbeddingELModel(Model):
    """Abstract class for :math:`\mathcal{EL}` embedding methods.

    :param dataset: mOWL dataset to use for training and evaluation.
    :type dataset: :class:`mowl.datasets.Dataset`
    :param embed_dim: The embedding dimension.
    :type embed_dim: int
    :param batch_size: The batch size to use for training.
    :type batch_size: int
    :param extended: If `True`, the model is supposed with 7 EL normal forms. This will be \
reflected on the :class:`DataLoaders` that will be generated and also the model must \
    contain 7 loss functions. If `False`, the model will work with 4 normal forms only, \
merging the 3 extra to their corresponding origin normal forms. Defaults to True
    :type extended: bool, optional
    :param load_normalized: If `True`, the ontology is assumed to be normalized and GCIs are extracted directly. Defaults to False.
    :type load_normalized: bool, optional
    :param device: The device to use for training. Defaults to "cpu".
    :type device: str, optional
    """

    def __init__(self, dataset, embed_dim, batch_size, extended=True, model_filepath=None,
                 load_normalized=False, device="cpu", learning_rate=0.001):
        super().__init__(dataset, model_filepath=model_filepath)

        if not isinstance(embed_dim, int):
            raise TypeError("Parameter 'embed_dim' must be of type int.")

        if not isinstance(batch_size, int):
            raise TypeError("Parameter batch_size must be of type int.")

        if not isinstance(extended, bool):
            raise TypeError("Optional parameter extended must be of type bool.")

        if not isinstance(load_normalized, bool):
            raise TypeError("Optional parameter load_normalized must be of type bool.")

        if not isinstance(device, str):
            raise TypeError("Optional parameter device must be of type str.")

        self._datasets_loaded = False
        self._dataloaders_loaded = False
        self._extended = extended
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.device = device
        self.load_normalized = load_normalized
        self.learning_rate = learning_rate

        self._training_datasets = None
        self._validation_datasets = None
        self._testing_datasets = None

        self._loaded_eval = False

    def init_module(self):
        raise NotImplementedError

    def _load_datasets(self):
        """This method will create different data attributes and finally the corresponding \
            DataLoaders for each GCI type in each subset (training, validation and testing).
        """
        if self._datasets_loaded:
            return

        # Get ontology paths for caching if available (PathDataset provides these)
        ontology_path = getattr(self.dataset, 'ontology_path', None)
        validation_path = getattr(self.dataset, 'validation_path', None)
        testing_path = getattr(self.dataset, 'testing_path', None)

        training_el_dataset = ELDataset(self.dataset.ontology,
                                        self.class_index_dict,
                                        self.object_property_index_dict,
                                        extended=self._extended,
                                        load_normalized=self.load_normalized,
                                        device=self.device,
                                        ontology_path=ontology_path)

        self._training_datasets = training_el_dataset.get_gci_datasets()

        self._validation_datasets = None
        if self.dataset.validation:
            validation_el_dataset = ELDataset(self.dataset.validation, self.class_index_dict,
                                              self.object_property_index_dict,
                                              extended=self._extended, device=self.device,
                                              ontology_path=validation_path)

            self._validation_datasets = validation_el_dataset.get_gci_datasets()

        self._testing_datasets = None
        if self.dataset.testing:
            testing_el_dataset = ELDataset(self.dataset.testing, self.class_index_dict,
                                           self.object_property_index_dict,
                                           extended=self._extended, device=self.device,
                                           ontology_path=testing_path)

            self._testing_datasets = testing_el_dataset.get_gci_datasets()

        self._datasets_loaded = True

    def _load_dataloaders(self):
        if self._dataloaders_loaded:
            return

        self._load_datasets()

        self._training_dataloaders = {
            k: DataLoader(v, batch_size=self.batch_size, pin_memory=False) for k, v in
            self._training_datasets.items()}

        if self._validation_datasets:
            self._validation_dataloaders = {
                k: DataLoader(v, batch_size=self.batch_size, pin_memory=False) for k, v in
                self._validation_datasets.items()}

        if self._testing_datasets:
            self._testing_dataloaders = {
                k: DataLoader(v, batch_size=self.batch_size, pin_memory=False) for k, v in
                self._testing_datasets.items()}

        self._dataloaders_loaded = True

    @property
    def training_datasets(self):
        """Returns the training datasets for each GCI type. Each dataset is an instance \
of :class:`mowl.datasets.el.ELDataset`

        :rtype: dict
        """
        self._load_datasets()
        return self._training_datasets

    @property
    def validation_datasets(self):
        """Returns the validation datasets for each GCI type. Each dataset is an instance \
of :class:`mowl.datasets.el.ELDataset`

        :rtype: dict
        """
        if self.dataset.validation is None:
            raise AttributeError("Validation dataset is None.")

        self._load_datasets()
        return self._validation_datasets

    @property
    def testing_datasets(self):
        """Returns the testing datasets for each GCI type. Each dataset is an instance \
of :class:`mowl.datasets.el.ELDataset`

        :rtype: dict
        """
        if self.dataset.testing is None:
            raise AttributeError("Testing dataset is None.")

        self._load_datasets()
        return self._testing_datasets

    @property
    def training_dataloaders(self):
        """Returns the training dataloaders for each GCI type. Each dataloader is an instance \
of :class:`torch.utils.data.DataLoader`

        :rtype: dict
        """
        self._load_dataloaders()
        return self._training_dataloaders

    @property
    def validation_dataloaders(self):
        """Returns the validation dataloaders for each GCI type. Each dataloader is an instance \
of :class:`torch.utils.data.DataLoader`

        :rtype: dict
        """
        if self.dataset.validation is None:
            raise AttributeError("Validation dataloader is None.")

        self._load_dataloaders()
        return self._validation_dataloaders

    @property
    def testing_dataloaders(self):
        """Returns the testing dataloaders for each GCI type. Each dataloader is an instance \
of :class:`torch.utils.data.DataLoader`

        :rtype: dict
        """
        if self.dataset.testing is None:
            raise AttributeError("Testing dataloader is None.")

        self._load_dataloaders()
        return self._testing_dataloaders

    # ==================== Training Methods ====================

    def get_negative_sampling_config(self):
        """Returns configuration for negative sampling.

        Override this method to customize which GCI types require negative sampling
        and how negatives should be generated.

        :return: Dictionary mapping GCI names to their negative sampling config.
            Each config should have:
            - 'index_pool': str, either 'classes' or 'individuals'
            - 'corrupt_column': int, which column to corrupt with random indices
        :rtype: dict

        Example::

            {
                "gci2": {"index_pool": "classes", "corrupt_column": 2},
                "object_property_assertion": {"index_pool": "individuals", "corrupt_column": 2}
            }
        """
        return {
            "gci2": {"index_pool": "classes", "corrupt_column": 2},
            "object_property_assertion": {"index_pool": "individuals", "corrupt_column": 2}
        }

    def generate_negatives(self, gci_name, gci_dataset):
        """Generate negative samples for a given GCI type.

        Override this method for custom negative sampling strategies.

        :param gci_name: Name of the GCI type (e.g., 'gci2')
        :type gci_name: str
        :param gci_dataset: The dataset containing positive samples
        :type gci_dataset: torch.Tensor
        :return: Negative samples tensor, or None if no negatives for this GCI type
        :rtype: torch.Tensor or None
        """
        config = self.get_negative_sampling_config()
        if gci_name not in config:
            return None

        cfg = config[gci_name]
        index_pool = cfg["index_pool"]
        corrupt_column = cfg["corrupt_column"]

        if index_pool == "classes":
            all_ids = list(self.class_index_dict.values())
        elif index_pool == "individuals":
            all_ids = list(self.individual_index_dict.values())
        else:
            raise ValueError(f"Unknown index_pool: {index_pool}")

        data = gci_dataset[:]
        idxs_for_negs = np.random.choice(all_ids, size=len(gci_dataset), replace=True)
        rand_index = th.tensor(idxs_for_negs, dtype=th.long, device=self.device)

        # Build negative data by replacing the specified column
        neg_data = th.cat([data[:, :corrupt_column], rand_index.unsqueeze(1)], dim=1)
        if corrupt_column + 1 < data.shape[1]:
            neg_data = th.cat([neg_data, data[:, corrupt_column + 1:]], dim=1)

        return neg_data

    def compute_loss(self, pos_scores, neg_scores=None):
        """Compute loss from positive and negative scores.

        Override this method to use different loss functions (e.g., MSE loss).

        :param pos_scores: Scores for positive samples (should be minimized)
        :type pos_scores: torch.Tensor
        :param neg_scores: Scores for negative samples (should be maximized), or None
        :type neg_scores: torch.Tensor or None
        :return: Combined loss value
        :rtype: torch.Tensor
        """
        loss = th.mean(pos_scores)
        if neg_scores is not None:
            loss += th.mean(neg_scores)
        return loss

    def get_regularization_loss(self):
        """Get regularization loss from the module.

        Override this method if your module has a regularization loss.

        :return: Regularization loss value
        :rtype: torch.Tensor
        """
        if hasattr(self.module, 'regularization_loss'):
            return self.module.regularization_loss()
        return 0

    def get_optimizer(self):
        """Create and return the optimizer.

        Override this method to use a different optimizer or configuration.

        :return: Optimizer instance
        :rtype: torch.optim.Optimizer
        """
        return th.optim.Adam(self.module.parameters(), lr=self.learning_rate)

    def train(self, epochs, validate_every=1):
        """Train the model.

        This is the generic training loop for EL embedding models. Subclasses can
        customize behavior by overriding:
        - :meth:`get_negative_sampling_config`: Configure which GCIs need negatives
        - :meth:`generate_negatives`: Custom negative sampling strategy
        - :meth:`compute_loss`: Custom loss computation (e.g., MSE loss)
        - :meth:`get_regularization_loss`: Add regularization
        - :meth:`get_optimizer`: Use different optimizer

        :param epochs: Number of training epochs
        :type epochs: int
        :param validate_every: Validate and log every N epochs. Defaults to 1.
        :type validate_every: int, optional
        """
        logger.warning(
            'You are using the default training method. If you want to use a customized '
            'training method (e.g., different negative sampling, etc.), please override '
            'the appropriate methods in a subclass.'
        )

        # Log dataset sizes
        points_per_dataset = {k: len(v) for k, v in self.training_datasets.items()}
        string = "Training datasets: \n"
        for k, v in points_per_dataset.items():
            string += f"\t{k}: {v}\n"
        logger.info(string)

        optimizer = self.get_optimizer()
        best_loss = float('inf')

        for epoch in trange(epochs):
            self.module.train()

            train_loss = 0
            loss = th.tensor(0.0, device=self.device)

            for gci_name, gci_dataset in self.training_datasets.items():
                if len(gci_dataset) == 0:
                    continue

                # Compute positive loss
                pos_scores = self.module(gci_dataset[:], gci_name)

                # Generate and compute negative loss if applicable
                neg_data = self.generate_negatives(gci_name, gci_dataset)
                neg_scores = None
                if neg_data is not None:
                    neg_scores = self.module(neg_data, gci_name, neg=True)

                loss = loss + self.compute_loss(pos_scores, neg_scores)

            # Add regularization loss
            loss = loss + self.get_regularization_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            # Validation
            if (epoch + 1) % validate_every == 0:
                if self.dataset.validation is not None:
                    with th.no_grad():
                        self.module.eval()
                        valid_loss = 0
                        gci2_data = self.validation_datasets["gci2"][:]
                        vloss = th.mean(self.module(gci2_data, "gci2"))
                        valid_loss += vloss.detach().item()

                        if valid_loss < best_loss:
                            best_loss = valid_loss
                            th.save(self.module.state_dict(), self.model_filepath)
                    print(f'Epoch {epoch+1}: Train loss: {train_loss} Valid loss: {valid_loss}')
                else:
                    print(f'Epoch {epoch+1}: Train loss: {train_loss}')

    def eval_method(self, data):
        """Evaluation method used for scoring. Override if needed.

        :param data: Input data for evaluation
        :type data: torch.Tensor
        :return: Evaluation scores
        :rtype: torch.Tensor
        """
        return self.module.gci2_loss(data)

    def get_embeddings(self):
        """Get trained embeddings for entities, relations, and individuals.

        :return: Tuple of (entity_embeddings, relation_embeddings, individual_embeddings)
        :rtype: tuple
        """
        self.init_module()
        print('Load the best model', self.model_filepath)
        self.load_best_model()

        ent_embeds = {
            k: v for k, v in zip(self.class_index_dict.keys(),
                                 self.module.class_embed.weight.cpu().detach().numpy())}
        rel_embeds = {
            k: v for k, v in zip(self.object_property_index_dict.keys(),
                                 self.module.rel_embed.weight.cpu().detach().numpy())}
        if self.module.ind_embed is not None:
            ind_embeds = {
                k: v for k, v in zip(self.individual_index_dict.keys(),
                                     self.module.ind_embed.weight.cpu().detach().numpy())}
        else:
            ind_embeds = None
        return ent_embeds, rel_embeds, ind_embeds

    def load_best_model(self):
        """Load the best model from the model filepath."""
        self.init_module()
        self.module.load_state_dict(th.load(self.model_filepath, weights_only=True))
        self.module.eval()

    @versionadded(version="0.2.0")
    def score(self, axiom):
        """
        Returns the score of the given axiom.

        :param axiom: The axiom to score.
        :type axiom: :class:`org.semanticweb.owlapi.model.OWLAxiom`
        """

        def data_point_to_tensor(data_point):
            data_point = th.tensor(data_point, dtype=th.long, device=self.device)
            data_point = data_point.unsqueeze(0)
            return data_point
        
        not_el_error_msg = "This axiom does not belong to the EL description logic specification."
        sub, super_ = axiom.getSubClass(), axiom.getSuperClass()

        if not isinstance(sub, OWLClassExpression):
            raise TypeError("Parameter sub must be of type OWLClassExpression.")

        if isinstance(sub, OWLClass):
            sub_id = self.dataset.class_to_id[sub]
            if isinstance(super_, OWLClass):
                super_id = self.dataset.class_to_id[super_]
                if super_.isOWLNothing():
                    if self.extended:
                        gci_name = "gci0_bot"
                    else:
                        gci_name = "gci0"
                else:
                    gci_name = "gci0"

                gci_data = data_point_to_tensor([sub_id, super_id])

            elif isinstance(super_, OWLObjectSomeValuesFrom):
                rel = super_.getProperty()
                filler = super_.getFiller()
                if not isinstance(filler, OWLClass):
                    raise TypeError(not_el_error_msg)
                
                rel_id = self.dataset.object_property_to_id[rel]
                filler_id = self.dataset.class_to_id[filler]
                gci_name = "gci2"
                gci_data = data_point_to_tensor([sub_id, rel_id, filler_id])
                
        elif isinstance(sub, OWLObjectSomeValuesFrom):
            rel = sub.getProperty()
            filler = sub.getFiller()
            if not isinstance(filler, OWLClass):
                raise TypeError(not_el_error_msg)
            if not isinstance(super_, OWLClass):
                raise TypeError(not_el_error_msg)

            rel_id = self.dataset.object_property_to_id[rel]
            filler_id = self.dataset.class_to_id[filler]
            super_id = self.dataset.class_to_id[super_]
            if super_.isOWLNothing():
                if self.extended:
                    gci_name = "gci3_bot"
                else:
                    gci_name = "gci3"
            else:
                gci_name = "gci3"
            
            gci_data = data_point_to_tensor([rel_id, filler_id, super_id])
            
        elif isinstance(sub, OWLObjectIntersectionOf):
            operands = sub.getOperandsAsList()
            if len(operands) != 2:
                raise TypeError(not_el_error_msg)
            left, right = tuple(operands)
            if not isinstance(left, OWLClass):
                raise TypeError(not_el_error_msg)
            if not isinstance(right, OWLClass):
                raise TypeError(not_el_error_msg)
            if not isinstance(super_, OWLClass):
                raise TypeError(not_el_error_msg)

            left_id = self.dataset.class_to_id[left]
            right_id = self.dataset.class_to_id[right]
            super_id = self.dataset.class_to_id[super_]

            if super_.isOWLNothing():
                if self.extended:
                    gci_name = "gci1_bot"
                else:
                    gci_name = "gci1"
            else:
                gci_name = "gci1"

            gci_data = data_point_to_tensor([left_id, right_id, super_id])
            
        else:
            raise TypeError("This axiom does not belong to EL.")

        
        score = self.module(gci_data, gci_name)
        return score


    @property
    def class_embeddings(self):
        class_embeds = {
            k: v for k, v in zip(self.class_index_dict.keys(),
                                 self.module.class_embed.weight.cpu().detach().numpy())}
        return class_embeds

    @property
    def object_property_embeddings(self):
        rel_embeds = {
            k: v for k, v in zip(self.object_property_index_dict.keys(),
                                 self.module.rel_embed.weight.cpu().detach().numpy())}
        
        return rel_embeds
        
    
    @property
    def individual_embeddings(self):
        if self.module.ind_embed is None:
            return dict()
        
        ind_embeds = {
            k: v for k, v in zip(self.individual_index_dict.keys(),
                                 self.module.ind_embed.weight.cpu().detach().numpy())}
        
        return ind_embeds
        


    def add_axioms(self, *axioms):
        prev_class_embeds = None
        prev_object_property_embeds = None
        prev_individual_embeds = None
        
        if len(self.class_embeddings) > 0:
            prev_class_embeds = copy.deepcopy(self.class_embeddings)

        if len(self.object_property_embeddings) > 0:
            prev_object_property_embeds = copy.deepcopy(self.object_property_embeddings)

        if len(self.individual_embeddings) > 0:
            prev_individual_embeds = copy.deepcopy(self.individual_embeddings)

        self.dataset.add_axioms(*axioms)

        if prev_class_embeds is not None:
            new_class_embeds = []
            for cls in self.dataset.classes:
                cls = str(cls.toStringID())
                if cls in prev_class_embeds:
                    new_class_embeds.append(prev_class_embeds[cls])
                else:
                    new_class_embeds.append(np.random.normal(size=self.embed_dim))
            

            new_class_embeds = np.asarray(new_class_embeds)
            self.module.class_embed.weight.data = th.from_numpy(new_class_embeds).float()

        if prev_object_property_embeds is not None:
            new_object_property_embeds = []
            for rel in self.dataset.object_properties:
                rel = str(rel.toStringID())
                if rel in prev_object_property_embeds:
                    new_object_property_embeds.append(prev_object_property_embeds[rel])
                else:
                    new_object_property_embeds.append(np.random.normal(size=self.embed_dim))
            
            new_object_property_embeds = np.asarray(new_object_property_embeds)
            self.module.rel_embed.weight.data = th.from_numpy(new_object_property_embeds).float()

        if prev_individual_embeds is not None:
            new_individual_embeds = []
            for ind in self.dataset.individuals:
                ind = str(ind.toStringID())
                if ind in prev_individual_embeds:
                    new_individual_embeds.append(prev_individual_embeds[ind])
                else:
                    new_individual_embeds.append(np.random.normal(size=self.embed_dim))
            
            new_individual_embeds = np.asarray(new_individual_embeds)
            self.module.ind_embed.weight.data = th.from_numpy(new_individual_embeds).float()

            

    def from_pretrained(self, model):
        if not isinstance(model, str):
            raise TypeError("Parameter model must be a string pointing to the model file.")

        if not os.path.exists(model):
            raise FileNotFoundError("Pretrained model path does not exist")

        #self._model_filepath = model

        
        self._is_pretrained = True
        if not isinstance(model, str):
            raise TypeError

        self.module.load_state_dict(th.load(model, weights_only=True))
        #self._kge_method = kge_method
    



    def load_pairwise_eval_data(self):

        if self._loaded_eval:
            return

        eval_property = self.dataset.get_evaluation_property()
        head_classes, tail_classes = self.dataset.evaluation_classes
        self._head_entities = head_classes.as_str
        self._tail_entities = tail_classes.as_str
                        
        eval_projector = projector_factory('taxonomy_rels', taxonomy=False,
                                           relations=[eval_property])

        self._training_set = eval_projector.project(self.dataset.ontology)
        self._testing_set = eval_projector.project(self.dataset.testing)

        self._loaded_eval = True


    @property
    def training_set(self):
        self.load_pairwise_eval_data()
        return self._training_set

    @property
    def testing_set(self):
        self.load_pairwise_eval_data()
        return self._testing_set

    @property
    def head_entities(self):
        self.load_pairwise_eval_data()
        return self._head_entities

    @property
    def tail_entities(self):
        self.load_pairwise_eval_data()
        return self._tail_entities
