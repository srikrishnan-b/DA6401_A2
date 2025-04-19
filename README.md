# DA6401 Assignment 2
Assignment 2 of DA6401 JanMay2025

Author: Srikrishnan B, BT23S013

### Description
This repository contains codes written as a part of Assignment 2 of DA6401 Jan-May 2025. The codes are written for the following tasks: training, hyperparameter tuning and finetuning of a pretrained model. The dataset used is `inaturalist12k`, which has 10000 images segregated into training and validation sets, covering 10 classes. 

### Requirements
The following packages are required:
- numpy   
- matplotlib
- scikit-learn  (for data splitting)
- torch
- torchvision
- pytorch_lightning
- wandb

An environment can be created using `requirements.txt`.

### Usage

The repository has notebooks to run the following tasks:
    - train and evaluate a CNN from scratch with a given configuration
        - visualize filters
        - plot predictions and images
    - perform hyperparameter tuning using wandb sweep functionality
    - finetuning a pretrained model (resnet50) on inaturalist dataset

The implementations are organized in script files. The notebooks utilize these functions and carry out the tasks.

- Use `notebooks/train_cnn.ipynb` for training. Hyperparameters and project names are set in `src/config.py`
- Use `notebooks/sweep.ipynb` for wand sweeps. Sweep configurations, project names are set in `src/sweep_config.py`
- use `notebooks/finetune.ipynb` for finetuning **resnet50**. 


### Folder organization

The codes are organized in two folders: `src` and `notebooks`. `src` contains all the source codes (.py files) and config files, and `notebooks` contains ipython notebooks that demonstrate training a CNN from scratch, performing a wandb sweep, and finetuning a pretrained model. 

```

│
├── README.md               # Documentation
├── requirements.txt        # Dependencies
│
├── src/                   
│   ├── config.py            # Configuration for training
│   ├── dataloader.py        # Load dataset, split and do transformations/augmentation
│   ├── label_map.py         # Mapping of labels to indices
│   ├── models.py            # All CNN implementations are defined here: scratch, lightning, finetuning etc
|   ├── sweep_config.py      # Configuration for sweep
│   ├── utils.py             # Visualize filters, plot predictions
├── notebooks/              
│   ├── finetune.ipynb       # Finetune resnet50
│   ├── sweep.ipynb          # Wandb sweep
│   └── train_cnn.ipynb      # Train CNN from scratch

```


- `models.py`: The following functions are defined: CNN, train_CNN (for sweep), pretrained model



### Link to wandb report: [Link](https://wandb.ai/deeplearn24/dla2-hypersweeps/reports/BT23S013-DA6401-Assignment-2--VmlldzoxMjI5OTUzMg?accessToken=bpcqz50a3d769f4n7xgc3wdyk8hspwqzluf2k79bee6sm1wqme9nd959bbdh7ikw)
### Link to Github repo: [Link](https://github.com/srikrishnan-b/DA6401_A2)
 