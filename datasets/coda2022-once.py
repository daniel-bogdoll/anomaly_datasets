from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class CODA2022ONCE(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = []
        self.img_labels = []
        self.resolutions = [(720, 1355)]
        self.ood_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for im in os.listdir(self.root): #TODO
            self.images.append(os.path.join(self.root, im)) #TODO
            self.img_labels.append(os.path.join(self.root, im)) #TODO

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        image = np.array(Image.open(self.images[i]))
        target = np.load(self.img_labels[i])
        return image, target
