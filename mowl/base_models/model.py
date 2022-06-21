class Model(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self._testing_set = None
    
    def train(self):
        '''Trains the model
        '''
        raise NotImplementedError()

    def eval_method():
        raise NotImplementedError()
    
    def evaluate(self):
        '''Evaluates the model
        '''
        raise NotImplementedError()


class EmbeddingModel(Model):

    def __init__(self, dataset):
        super().__init__(dataset)
        
        
    def get_entities_index_dict(self):
        raise NotImplementedError()
        

    def get_embeddings_data(self):
        raise NotImplementedError()



class EmbeddingELModel(EmbeddingModel):

    def __init__(self, dataset):
        super().__init__(dataset)

        self._data_loaded = False
        
    def gci0_loss(self):
        raise NotImplementedError()

    def gci0_bot_loss(self):
        raise NotImplementedError()

    def gci1_loss(self):
        raise NotImplementedError()

    def gci1_bot_loss(self):
        raise NotImplementedError()

    def gci2_loss(self):
        raise NotImplementedError()

    def gci3_loss(self):
        raise NotImplementedError()

    def gci3_bot_loss(self):
        raise NotImplementedError()


    def load_data(self):
        if self._data_loaded:
            return

        normalizer = ELNormalizer()
        all_axioms = []
        self.training_axioms = normalizer.normalize(self.dataset.ontology)
        all_axioms.append(self.training_axioms)
        
        if not self.dataset.validation is None:
            self.validation_axioms = normalizer.normalize(self.dataset.validation)
            all_axioms.append(self.validation_axioms)
            
        if not self.dataset.testing is None:
            self.testing_axioms = normalizer.normalize(self.dataset.testing)
            all_axioms.append(self.testing_axioms)
            
        classes = set()
        relations = set()

        for axioms_dict in all_axioms:
            for axiom in axioms_dict["gci0"]:
                classes.add(axiom.subclass)
                classes.add(axiom.superclass)

            for axiom in axioms_dict["gci0_bot"]:
                classes.add(axiom.subclass)
                classes.add(axiom.superclass)
            
            for axiom in axioms_dict["gci1"]:
                classes.add(axiom.left_subclass)
                classes.add(axiom.right_subclass)
                classes.add(axiom.superclass)

            for axiom in axioms_dict["gci1_bot"]:
                classes.add(axiom.left_subclass)
                classes.add(axiom.right_subclass)
                classes.add(axiom.superclass)

            for axiom in axioms_dict["gci2"]:
                classes.add(axiom.subclass)
                classes.add(axiom.filler)
                relations.add(axiom.obj_property)

            for axiom in axioms_dict["gci3"]:
                classes.add(axiom.superclass)
                classes.add(axiom.filler)
                relations.add(axiom.obj_property)

            for axiom in axioms_dict["gci3_bot"]:
                classes.add(axiom.superclass)
                classes.add(axiom.filler)
                relations.add(axiom.obj_property)

        self.classes_index_dict = {v: k  for k, v in enumerate(list(classes))}
        self.relations_index_dict = {v: k for k, v in enumerate(list(relations))}

        training_nfs = self.load_normal_forms(self.training_axioms, self.classes_index_dict, self.relations_index_dict)
        validation_nfs = self.load_normal_forms(self.validation_axioms, self.classes_index_dict, self.relations_index_dict)
        testing_nfs = self.load_normal_forms(self.testing_axioms, self.classes_index_dict, self.relations_index_dict)
        
        self.train_nfs = self.gcis_to_tensors(training_nfs, self.device)
        self.valid_nfs = self.gcis_to_tensors(validation_nfs, self.device)
        self.test_nfs = self.gcis_to_tensors(testing_nfs, self.device)
        self._loaded = True

        
    def load_normal_forms(self, axioms_dict, classes_dict, relations_dict):
        gci0 =     []
        gci0_bot = []
        gci1 =     []
        gci1_bot = []
        gci2 =     []
        gci3 =     []
        gci3_bot = []

        for axiom in axioms_dict["gci0"]:
            cl1 = classes_dict[axiom.subclass]
            cl2 = classes_dict[axiom.superclass]
            gci0.append((cl1, cl2))

        for axiom in axioms_dict["gci0_bot"]:
            cl1 = classes_dict[axiom.subclass]
            cl2 = classes_dict[axiom.superclass]
            gci0_bot.append((cl1, cl2))

        for axiom in axioms_dict["gci1"]:
            cl1 = classes_dict[axiom.left_subclass]
            cl2 = classes_dict[axiom.right_subclass]
            cl3 = classes_dict[axiom.superclass]
            gci1.append((cl1, cl2, cl3))

        for axiom in axioms_dict["gci1_bot"]:
            cl1 = classes_dict[axiom.left_subclass]
            cl2 = classes_dict[axiom.right_subclass]
            cl3 = classes_dict[axiom.superclass]
            gci1_bot.append((cl1, cl2, cl3))

        for axiom in axioms_dict["gci2"]:
            cl1 = classes_dict[axiom.subclass]
            rel = relations_dict[axiom.obj_property]
            cl2 = classes_dict[axiom.filler]
            gci2.append((cl1, rel, cl2))
        
        for axiom in axioms_dict["gci3"]:
            rel = relations_dict[axiom.obj_property]
            cl1 = classes_dict[axiom.filler]
            cl2 = classes_dict[axiom.superclass]
            gci3.append((rel, cl1, cl2))

        for axiom in axioms_dict["gci3_bot"]:
            rel = relations_dict[axiom.obj_property]
            cl1 = classes_dict[axiom.filler]
            cl2 = classes_dict[axiom.superclass]
            gci3_bot.append((rel, cl1, cl2))

        return gci0, gci0_bot, gci1, gci1_bot, gci2, gci3, gci3_bot
        
    def gcis_to_tensors(self, gcis, device):
        gci0, gci0_bot, gci1, gci1_bot, gci2, gci3, gci3_bot = gcis

        gci0     = th.LongTensor(gci0).to(device)
        gci0_bot = th.LongTensor(gci0_bot).to(device)
        gci1     = th.LongTensor(gci1).to(device)
        gci1_bot = th.LongTensor(gci1_bot).to(device)
        gci2     = th.LongTensor(gci2).to(device)
        gci3     = th.LongTensor(gci3).to(device)
        gci3_bot = th.LongTensor(gci3_bot).to(device)
        gcis = gci0, gci0_bot, gci1, gci1_bot, gci2, gci3, gci3_bot
        return gcis

