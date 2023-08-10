import lightning as L
from datasets import dataset_extr_to_mano
import glob
import json
import numpy as np
from torch.utils.data import DataLoader
import os

class MyDatamodule(L.LightningDataModule):
    def __init__(self, data_path, bg_rm_dilation):
        super().__init__()
        self.batch_size = 1
        self.data_path = data_path
        self.bg_rm_dilation = bg_rm_dilation

    def setup(self, stage=None):
        all_ids = list(range(len(
            glob.glob(os.path.join(self.data_path, 'images', '*.png'))
        )))

        # use 80% of the data for training, randomize the order
        np.random.shuffle(all_ids)
        train_ids = all_ids[int(0.3 * len(all_ids)):]
        test_ids = all_ids[:int(0.3 * len(all_ids))]

        self.train_dataset = dataset_extr_to_mano.NeumanDataset(self.data_path, train_ids, self.bg_rm_dilation)
        self.test_dataset = dataset_extr_to_mano.NeumanDataset(self.data_path, test_ids, self.bg_rm_dilation)
        self.full_dataset = dataset_extr_to_mano.NeumanDataset(self.data_path, all_ids, self.bg_rm_dilation)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)

    def val_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=False, num_workers=5)

