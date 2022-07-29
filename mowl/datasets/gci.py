from torch.utils.data import IterableDataset, Dataset

class GCIDataset(Dataset):
    def __init__(self, data, class_index_dict, object_property_index_dict = None, device = "cpu"):
        self.class_index_dict = class_index_dict
        self.object_property_index_dict = object_property_index_dict
        self.device = device
        self.data = self.push_to_device(data)
        
    def push_to_device(self):
        raise NotImplementedError()
    
    def get_data(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        return self.data[idx]
        
#    def __iter__(self):
 #       return self.get_data()

    def __len__(self):
        return len(self.data)

