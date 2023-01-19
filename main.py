import os
import sys
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

from utils import get_dataset, compute_heatmap


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):
    
    data = get_dataset(cfg[cfg.dataset])
    resolutions = data.resolutions
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=10)

    save_name = os.path.join(cfg.save_dir, 'size-' + cfg.dataset + '_max.png')

    compute_heatmap(loader, resolutions, data.ood_id, save_name)

if __name__ == '__main__':
    sys.argv.append('hydra.run.dir=./')
    main()