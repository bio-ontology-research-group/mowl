from torch.utils.data import IterableDataset, Dataset
import torch as th

class GCIDataset(Dataset):
    def __init__(self, data, class_index_dict, object_property_index_dict = None, device = "cpu"):
        super().__init__()
        self.class_index_dict = class_index_dict
        self.object_property_index_dict = object_property_index_dict
        self.device = device
        self._data = self.push_to_device(data)


    @property
    def data(self):
        return self._data

    
    def push_to_device(self):
        raise NotImplementedError()
    
    def get_data(self):
        raise NotImplementedError()

    def extend_from_indices(self, other):
        if isinstance(other, list):
            tensor = th.tensor(other, device = self.device)
        elif th.is_tensor(other):
            tensor = other
        else:
            raise TypeError("Extending element must be either a list or a Pytorch tensor.")

        assert self.data.shape[1:] == tensor.shape[1:], "Tensors must have the same shape except in the first dimension."

        tensor_elems = th.unique(tensor, return_counts = False, sorted = True)
        in_indices = sum(tensor_elems==i for i in list(self.class_index_dict.values())).bool()


        if not all(in_indices):
            raise ValueError("Extending element contains not recognized index.")
        
        new_tensor = th.cat([self._data, tensor.to(self.device)], dim = 0)
        self._data = new_tensor
        
    def __getitem__(self, idx):
        return self.data[idx]
        
#    def __iter__(self):
 #       return self.get_data()

    def __len__(self):
        return len(self.data)

