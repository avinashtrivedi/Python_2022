import numpy as np
import torch.utils.data as data


class Dataset(data.Dataset):

    def __init__(self, data_path):
        data = np.load(open(data_path, 'rb'))
        self.x = data['x'].astype(np.float32)
        self.y = data['y'].astype(np.int32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TestDataset(object):

    def __init__(self, data_path):
        data = np.load(open(data_path, 'rb'))
        self.x = data['x'].astype(np.float32)
        self.y = data['y'].astype(np.int32)

    def get_test_data(self):
        return self.x, self.y