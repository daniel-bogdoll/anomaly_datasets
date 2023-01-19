from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class FLAF(Dataset):
    def __init__(self, root = '/home/uhlemeyer/PyCharmProjects/dataset-obstacles-master/datasets/dataset_FishyLAF/labels/val'):
        self.root = root
        self.img_labels = []
        self.resolutions = [(1024,2048)]
        self.ood_id = [1]
        
        for im in os.listdir(self.root):
            self.img_labels.append(os.path.join(self.root, im))
                

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        target = Image.open(self.img_labels[i]).convert('L')
        return np.asarray(target, dtype=np.int16)
