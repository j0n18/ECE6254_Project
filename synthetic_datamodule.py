import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils import sample_annulus
import torch

class SyntheticDataModule(pl.LightningDataModule):

    def __init__(self, 
    dim: int,
    n_samples: int, 
    r_outer1: float, 
    r_inner1: float, 
    r_outer2: float, 
    r_inner2: float, 
    seed: int,
    batch_size: int,
    #num_workers: int = 4
    ):

        super().__init__()
        self.save_hyperparameters()


    def prepare_data(self):

        hps = self.hparams

        data = np.zeros( (2 * hps.n_samples, hps.dim) )

        #calculate the data for the annulus (data separable by circles; in R^6)
        x1, y1 = sample_annulus(hps.n_samples, hps.r_outer1, hps.r_inner1)
        x2, y2 = sample_annulus(hps.n_samples, hps.r_outer2, hps.r_inner2)

        #Assign x1,y1 to label 1, and x2,y2 to label 2:
        labels1 = np.zeros( (x1.shape[0], 1) )
        labels2 = np.ones( (x1.shape[0], 1) )

        x = np.concatenate((x1,x2))
        y = np.concatenate((y1,y2))
        labels = np.concatenate((labels1,labels2))

        labels = labels.squeeze()
        labels = labels.squeeze()

        data[:, 0] = x
        data[:, 1] = y

        # Perform data splits
        inds = np.arange(2 * hps.n_samples)
        train_inds, test_inds = train_test_split(
            inds, test_size=0.2, random_state=hps.seed
        )
        train_inds, valid_inds = train_test_split(
            train_inds, test_size=0.2, random_state=hps.seed
        )

        # sort the inds for easy retrieval later on:
        train_inds = np.sort(train_inds)
        valid_inds = np.sort(valid_inds)
        test_inds = np.sort(test_inds)

        #generate datasets from inds:
        train_data = data[train_inds]
        valid_data = data[valid_inds]
        test_data = data[test_inds]

        train_labels = labels[train_inds]
        valid_labels = labels[valid_inds]
        test_labels = labels[test_inds]


        #save datasets to object (as tensors):
        self.train_data = torch.tensor(train_data, dtype=torch.float)
        self.valid_data = torch.tensor(valid_data, dtype=torch.float)
        self.test_data = torch.tensor(test_data, dtype=torch.float)

        #save labels to object:
        self.train_labels = torch.tensor(train_labels, dtype=torch.float)
        self.valid_labels = torch.tensor(valid_labels, dtype=torch.float)
        self.test_labels = torch.tensor(test_labels, dtype=torch.float)

        #save indices to object:
        self.train_inds = torch.tensor(train_inds, dtype=torch.float)
        self.valid_inds = torch.tensor(valid_inds, dtype=torch.float)
        self.test_inds = torch.tensor(test_inds, dtype=torch.float)


    def setup(self, stage = None):
        self.train_ds = TensorDataset(self.train_data, self.train_labels, self.train_inds)
        self.valid_ds = TensorDataset(self.valid_data, self.valid_labels, self.valid_inds)
        self.test_ds = TensorDataset(self.test_data, self.test_labels, self.test_inds)


    def train_dataloader(self, shuffle = True):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            #num_workers = self.hparams.num_workers,
            shuffle=shuffle,
        )

        return train_dl


    def val_dataloader(self, shuffle = True):
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            #num_workers = self.hparams.num_workers,
            shuffle=shuffle,
        )

        return valid_dl
