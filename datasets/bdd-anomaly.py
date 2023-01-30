from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class BDDAnomaly(Dataset):
    def __init__(self, root="/datasets/anomaly-seg/seg/labels"):
        self.root = root
        self.img_labels = []
        self.resolutions = [(720, 1280)]
        self.ood_id = [16, 17]

        for split in ["val", "train"]:
            for im in os.listdir(os.path.join(self.root, split)):
                self.img_labels.append(os.path.join(self.root, split, im))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        target = Image.open(self.img_labels[i]).convert("L")
        return np.asarray(target, dtype=np.int16)
