from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class SH(Dataset):
    def __init__(self, root = '/home/datasets/streethazards/test/annotations/test'):
        self.root = root
        self.img_labels = []
        self.resolutions = [(720, 1280)]
        self.ood_id = [14]
        
        for split in os.listdir(self.root):
            for im in os.listdir(os.path.join(self.root, split)):
                self.img_labels.append(os.path.join(self.root, split, im))
                

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        target = Image.open(self.img_labels[i]).convert('L')
        return np.asarray(target, dtype=np.int16)
