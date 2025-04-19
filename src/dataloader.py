# Loading the required libraries
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision
import torchvision.transforms.v2 as v2
import os
import numpy as np
import random

g = torch.Generator()
g.manual_seed(42)


# Functions for augmentation / transformation
# ============================================================================================================
# Augmentation class
class Augmentation(torch.utils.data.Dataset):
    def __init__(self, train_complete, indices, transform):
        self.train_complete = train_complete
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, label = self.train_complete[actual_idx]
        if self.transform:
            image = self.transform(image)
        return image, label

#=============================================================================================================
# Loading dataset
#=============================================================================================================
class iNat_dataset:
    def __init__(self, data_dir, augmentation, batch_size, NUM_WORKERS=0):
        self.DARA_DIR = data_dir
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.NUM_WORKERS = NUM_WORKERS

    def load_dataset(self):

        # Defining transforms
        train_transform = self.transforms(augmentation=self.augmentation)
        val_transform = self.transforms(augmentation=False)
        test_transform = self.transforms(augmentation=False)

        train_dataset_complete = torchvision.datasets.ImageFolder(
            root=os.path.join(self.DATA_DIR, "train")
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.DATA_DIR, "val", transform=test_transform)
        )

        # Getting labels and random splitting/shuffling of each class examples
        labels = np.array([entry[1] for entry in train_dataset_complete.samples])
        split_fn = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=219)
        train_ids, valid_ids = next(split_fn.split(np.zeros(len(labels)), labels))

        # Transforms for train and valid sets (no augmentation in validation set)
        train_dataset = Augmentation(train_dataset_complete, train_ids, train_transform)
        valid_dataset = Augmentation(train_dataset_complete, valid_ids, val_transform)

        # Dataloader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.NUM_WORKERS,
            pin_memory=False,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        val_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.NUM_WORKERS,
            pin_memory=False,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.NUM_WORKERS,
            pin_memory=False,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        classes = train_dataset_complete.classes
        n_classes = len(classes)

        return train_dataloader, val_dataloader, test_dataloader, classes, n_classes

    # Function to apply transformations to the dataset
    def transforms(augmentation):
        if augmentation:
            transform = v2.Compose(
                [
                    v2.Resize((224, 224)),
                    v2.RandomHorizontalFlip(p=0.4),
                    v2.RandomVerticalFlip(p=0.1),
                    v2.RandomApply([v2.RandomRotation(degrees=15)], p=0.1),
                    v2.RandomApply(
                        [
                            v2.ColorJitter(
                                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                            )
                        ],
                        p=0.5,
                    ),
                    
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            transform = v2.Compose(
                [
                    v2.Resize((224, 224)),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

        return transform

    # For torch dataloader
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


#======================================================================================================