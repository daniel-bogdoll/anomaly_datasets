from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class CODAONCE(Dataset):
    def __init__(self, root="/datasets/CODA_masks/once"):
        self.root = root
        self.img_labels = []
        self.resolutions = [(1020, 1920)]
        self.ood_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for im in os.listdir(self.root):
            self.img_labels.append(os.path.join(self.root, im))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        target = np.load(self.img_labels[i])
        return target
