from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class FST(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = []
        self.img_labels = []
        self.resolutions = [(1024, 2048)]
        self.ood_id = [1]

        for im in os.listdir(self.root):
            if im[-3:] == "png":
                self.img_labels.append(os.path.join(self.root, im))
            elif im[-3:] == "jpg":
                self.images.append(os.path.join(self.root, im))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        image = np.array(Image.open(self.images[i]))
        target = Image.open(self.img_labels[i]).convert("L")
        target = np.asarray(target, dtype=np.int16)
        return image, target
