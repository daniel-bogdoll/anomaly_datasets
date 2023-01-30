from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class LAF(Dataset):
    def __init__(self, root="/datasets/lost_and_found/gtCoarse/test"):
        self.root = root
        self.img_labels = []
        self.resolutions = [(1024, 2048)]
        self.ood_id = np.arange(2, 44)

        for street in os.listdir(self.root):
            for im in os.listdir(os.path.join(self.root, street)):
                if im.split("_")[-1] == "labelTrainIds.png":
                    self.img_labels.append(os.path.join(self.root, street, im))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        target = Image.open(self.img_labels[i]).convert("L")
        return np.asarray(target, dtype=np.int16)
