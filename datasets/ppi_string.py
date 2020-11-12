from datasets.base import Dataset

class PPI_String(Dataset):

    def __init__(self, url, *args, *kwargs):
        super(PPI_String, self).__init__(url, *args, **kwargs)

    def _load(self):
        if self._loaded:
            return
        # TODO: implement loading here
        self._loaded = True

