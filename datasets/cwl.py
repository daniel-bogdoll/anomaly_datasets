from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class CWL(Dataset):
    def __init__(self, root="/datasets/carla_wildlife_sequences/semantic_ood"):
        self.root = root
        self.img_labels = []
        self.resolutions = [(1080, 1920)]
        self.ood_id = [254]

        for seq in os.listdir(root):
            for im in os.listdir(os.path.join(self.root, seq)):
                self.img_labels.append(os.path.join(self.root, seq, im))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        target = Image.open(self.img_labels[i]).convert("L")
        return np.asarray(target, dtype=np.int16)
