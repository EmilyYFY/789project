import numpy as np
import torch
from torch.utils.data import Dataset
import os

#build the dataset from png
class ShapeDataset(Dataset):
    def __init__(self, npz_file):
        self.data=np.load(npz_file)
        self.images=self.data['images']
        self.labels=self.data['labels']
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image=self.images[idx,:,:].astype(np.float32)
        label=self.labels[idx]
        image=np.expand_dims(image, axis=0)
        label=np.expand_dims(label, axis=0)
        return image,label