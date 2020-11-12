class Dataset(object):
    def __init__(self, url, *args, **kwargs):
        self.url = url

    def _load(self):
        raise NotImplementedError()
