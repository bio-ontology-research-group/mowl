from tests.datasetFactory import FamilyDataset
from mowl.owlapi import OWLAPIAdapter
import random
from mowl.owlapi.defaults import BOT
import torch as th

class ELAxioms():

    def __init__(self):
        adapter = OWLAPIAdapter()
        bot_class = adapter.create_class(BOT)
        self.dataset = FamilyDataset()

        self.classes = self.dataset.classes
        self.properties = self.dataset.object_properties

        class_1 = random.choice(self.classes)
        class_2 = random.choice(self.classes)
        class_3 = random.choice(self.classes)
        class_4 = random.choice(self.classes)
        relation_1 = random.choice(self.properties)

        class_1_id = self.dataset.class_to_id[class_1]
        class_2_id = self.dataset.class_to_id[class_2]
        class_3_id = self.dataset.class_to_id[class_3]
        class_4_id = self.dataset.class_to_id[class_4]
        bot_id = self.dataset.class_to_id[bot_class]
        relation_1_id = self.dataset.object_property_to_id[relation_1]

        self.gci0_data = th.LongTensor([[class_1_id, class_2_id]])
        self.gci1_data = th.LongTensor([[class_1_id, class_2_id, class_3_id]])
        self.gci2_data = th.LongTensor([[class_1_id, relation_1_id, class_2_id]])
        self.gci3_data = th.LongTensor([[relation_1_id, class_1_id, class_2_id]])
        self.gci0_bot_data = th.LongTensor([[class_1_id, bot_id]])
        self.gci1_bot_data = th.LongTensor([[class_1_id, class_2_id, bot_id]])
        self.gci3_bot_data = th.LongTensor([[relation_1_id, class_1_id, bot_id]])
        
