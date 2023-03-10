from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class VistasNP(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = []
        self.img_labels = []
        self.resolutions = [
            (3000, 4000),
            (2449, 3265),
            (1836, 3264),
            (2448, 3264),
            (2989, 3985),
            (2592, 4608),
            (2160, 3840),
            (2988, 3984),
            (1944, 2592),
            (3024, 4032),
            (1536, 2048),
            (1341, 2012),
            (3888, 5152),
            (1080, 1920),
            (3456, 4608),
            (1200, 1600),
            (1920, 2560),
            (1067, 1600),
            (1537, 2048),
            (3096, 4128),
            (2250, 4000),
            (1936, 2592),
            (3936, 5248),
            (2250, 3000),
            (720, 1280),
            (3120, 4160),
            (1296, 2304),
            (2637, 4032),
            (1968, 2624),
            (2640, 3520),
            (1512, 2688),
            (2400, 3200),
            (768, 1024),
            (2048, 2048),
            (2424, 3232),
            (2760, 3680),
            (2443, 4000),
            (3025, 4033),
            (3000, 4496),
            (2048, 2448),
            (2988, 5312),
            (1456, 2592),
            (1810, 3580),
            (960, 1280),
            (1520, 2688),
            (2016, 3024),
            (3030, 4546),
            (2304, 4096),
            (1342, 2013),
            (2976, 2976),
            (1500, 2000),
            (2160, 4096),
            (1440, 1920),
            (2736, 3648),
            (2268, 4032),
            (1728, 1728),
            (1937, 2593),
            (1945, 2593),
            (1840, 3264),
            (3121, 4160),
            (1184, 1600),
            (2304, 3840),
            (1800, 2400),
            (1537, 2560),
            (1440, 2560),
            (1968, 3264),
            (3752, 5376),
            (1524, 2704),
            (2000, 1500),
            (1744, 3104),
            (2414, 4608),
            (1960, 2608),
            (3456, 5184),
            (2086, 4608),
            (2448, 2448),
            (2322, 4128),
            (2977, 2977),
            (3072, 4096),
            (1536, 2560),
            (4128, 3096),
            (2256, 3008),
            (2592, 3456),
            (900, 1355),
            (1200, 1800),
            (3120, 4208),
            (3984, 2988),
            (2240, 4000),
            (3648, 2736),
            (2160, 3600),
            (3680, 6528),
            (2000, 3552),
            (1552, 2592),
            (3097, 4128),
            (1458, 2592),
            (1932, 2576),
            (3104, 4192),
            (1536, 2304),
            (3072, 4608),
            (1872, 3328),
            (600, 800),
            (3168, 4224),
            (1836, 2448),
            (1728, 2880),
            (2240, 3968),
            (2160, 2880),
            (1151, 1534),
            (5248, 3936),
            (1071, 1600),
            (3456, 4434),
            (4240, 5664),
            (1520, 2048),
            (2335, 4608),
            (1728, 2304),
            (2752, 4128),
            (3060, 4074),
            (2709, 4900),
            (1234, 1624),
            (2341, 3121),
            (2332, 4128),
            (2304, 3456),
            (2933, 4032),
            (1921, 2561),
            (2289, 3052),
            (2214, 4274),
            (1791, 4608),
            (1080, 1440),
            (1837, 3264),
            (2060, 2747),
            (2368, 3200),
            (1692, 2256),
            (1800, 3200),
            (1509, 2012),
            (4208, 3120),
            (2050, 2448),
            (5184, 3456),
            (2204, 3920),
            (2109, 3624),
            (1752, 4608),
            (3264, 2448),
            (4129, 3097),
            (3264, 1836),
            (2040, 2720),
            (2237, 4608),
            (806, 1400),
            (2365, 3153),
            (1728, 3072),
            (2432, 4320),
            (480, 640),
            (2176, 3264),
            (702, 1280),
            (2136, 4590),
            (2976, 3968),
            (1000, 750),
            (2326, 4608),
            (2150, 2873),
            (1750, 2333),
            (3744, 5376),
            (2431, 4608),
            (2500, 4128),
            (2316, 4608),
            (3136, 4224),
            (1140, 1521),
            (2160, 3920),
        ]

        person_color = [220, 20, 60]
        rider1_color = [255, 0, 0]
        rider2_color = [255, 0, 100]
        rider3_color = [255, 0, 200]

        # L = R * 299/1000 + G * 587/1000 + B * 114/1000 (https://pillow.readthedocs.io/en/stable/reference/Image.html)
        ood_1 = int(person_color[0] * 299 / 1000 + person_color[1] * 587 / 1000 + person_color[2] * 114 / 1000)
        ood_2 = int(rider1_color[0] * 299 / 1000 + rider1_color[1] * 587 / 1000 + rider1_color[2] * 114 / 1000)
        ood_3 = int(rider2_color[0] * 299 / 1000 + rider2_color[1] * 587 / 1000 + rider2_color[2] * 114 / 1000)
        ood_4 = int(rider3_color[0] * 299 / 1000 + rider3_color[1] * 587 / 1000 + rider3_color[2] * 114 / 1000)

        self.ood_id = [ood_1, ood_2, ood_3, ood_4]

        for split in ["test/labels", "training/labels", "validation/labels"]:
            for im in os.listdir(os.path.join(self.root, split)):
                self.images.append() #TODO
                self.img_labels.append(os.path.join(self.root, split, im))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        image = np.array(Image.open(self.images[i]))
        target = Image.open(self.img_labels[i]).convert("L")
        target = np.asarray(target, dtype=np.int16)
        return image, target
