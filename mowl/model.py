
class Model(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def train(self):
        raise NotImplementedError()
    
    def evaluate(self):
        raise NotImplementedError()
