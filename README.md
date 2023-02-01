# anomaly_datasets
This repository can be used to generate the cumulated masks as in our paper "Perception Datasets for Anomaly Detection in Autonomous Driving: A Survey"

These provide an overview of how many anomalies are contained in a dataset and in which regions they can be found.

## Structure
```bash
├── datasets                  # Configurations for individual datasets
├── figures                   # Output
| ├──── overlays              # Images overlaid with ground truth
| ├──── pixel-distributions   # Accumulated anomaly masks    
└── helper                    # Helper functions for CODA, Vistas-NP, and WD-Pascal
```

## Installation
It is recommended to create a python3 environment and install all required packages. Our code runs with python version 3.6
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Code
Before the code can be launched, the root paths to the datasets must be specified in config.yaml. 
Then you need to define for which dataset you want to create the cumulated masks. For this you need to change the dataset variable in config.yaml and run the following code: 
```bash
python main.py
```
Hydra allows overwriting config values in the terminal. This can be done as below using the CWL dataset as an example:
```bash
python main.py ++dataset=cwl
```

## Add a new dataset
If you want to add a dataset, you have to define your own class under the datasets folder. Feel free to use datasets/template.py for this purpose.

After that, another entry must be added to the configuration file according to the following scheme: 
```bash
new_dataset:
  _target_: datasets.new_dataset.ND
  root: /PATH/TO/DATASET
```

## Citation
If you find our work useful for your research, please cite our paper:
```
@article{Bogdoll_Perception_2023_arXiv,
    author    = {Daniel Bogdoll, Svenja Uhlemeyer, Kamil Kowol, J. Marius Zöllner},
    title     = {{Perception Datasets for Anomaly Detection in Autonomous Driving: A Survey}},
    journal   = {arXiv:XXXX.XXXXXXXX},
    year      = {2023}
}
```
