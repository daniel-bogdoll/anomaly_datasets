from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class WDP(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = []
        self.img_labels = []
        self.resolutions = [(1088, 1920)]
        self.ood_id = [1]

        for im in os.listdir(self.root, 'images'): #TODO
            self.images.append(os.path.join(self.root, 'images', im)) #TODO
            self.img_labels.append(os.path.join(self.root, 'labels', im)) #TODO

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        image = np.array(Image.open(self.images[i]))
        target = np.load(self.img_labels[i])
        return image, target
