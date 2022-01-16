class Model(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def train(self):
        '''Trains the model
        '''
        raise NotImplementedError()
    
    def evaluate(self):
        '''Evaluates the model
        '''
        raise NotImplementedError()


