from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import cv2

class RO21(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = []
        self.img_labels = []
        self.resolutions = [(1080,1920)]
        self.ood_id = [1]
        
        for im in os.listdir(os.path.join(self.root, 'images')):
            self.images.append(os.path.join(self.root, 'images', im))
            self.img_labels.append(os.path.join(self.root, 'labels_masks', im.replace('.webp', '_labels_semantic.png')))                

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        image = cv2.imread(self.images[i])
        target = Image.open(self.img_labels[i]).convert('L')
        target = np.asarray(target, dtype=np.int16)
        return image, target
