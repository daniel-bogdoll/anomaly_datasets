from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class CWL(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = []
        self.img_labels = []
        self.resolutions = [(1080, 1920)]
        self.ood_id = [254]

        for seq in os.listdir(os.path.join(root, 'raw_data')):
            for im in os.listdir(os.path.join(self.root, 'raw_data', seq)):
                self.images.append(os.path.join(root, 'raw_data', seq, im))
                self.img_labels.append(os.path.join(root, 'semantic_ood', seq, im.replace('raw_data.png', 'semantic_ood.png')))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        image = np.array(Image.open(self.images[i]))
        target = Image.open(self.img_labels[i]).convert("L")
        target = np.asarray(target, dtype=np.int16)
        return image, target
