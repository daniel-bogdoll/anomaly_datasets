from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class SH(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = []
        self.img_labels = []
        self.resolutions = [(720, 1280)]
        self.ood_id = [14]

        for split in os.listdir(os.path.join(self.root, 'images', 'test')):
            for im in os.listdir(os.path.join(self.root, 'images', 'test', split)):
                self.images.append(os.path.join(self.root, 'images', 'test', split, im))
                self.img_labels.append(os.path.join(self.root, 'annotations', 'test', split, im))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        image = np.array(Image.open(self.images[i]))
        target = Image.open(self.img_labels[i]).convert("L")
        target = np.asarray(target, dtype=np.int16)
        return image, target
