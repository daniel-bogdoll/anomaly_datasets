from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class RO21(Dataset):
    def __init__(self, root="/datasets/dataset_ObstacleTrack/labels_masks"):
        self.root = root
        self.img_labels = []
        self.resolutions = [(1080, 1920)]
        self.ood_id = [1]

        for im in os.listdir(self.root):
            if im.split("_")[-1] == "semantic.png":
                self.img_labels.append(os.path.join(self.root, im))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        target = Image.open(self.img_labels[i]).convert("L")
        return np.asarray(target, dtype=np.int16)
