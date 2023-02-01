from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class ND(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = []
        self.img_labels = []
        self.resolutions = []   # insert resolutions
        self.ood_id = []        # insert OoD ids

        for seq in os.listdir(root):
            for im in os.listdir(os.path.join(self.root, seq, 'images')):
                self.images.append(os.path.join(self.root, seq, im))
            for im in os.listdir(os.path.join(self.root, seq, 'label')):
                self.img_labels.append(os.path.join(self.root, seq, im))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        image = np.array(Image.open(self.images[i]))
        target = Image.open(self.img_labels[i]).convert("L")
        target = np.asarray(target, dtype=np.int16)
        return image, target
