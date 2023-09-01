import lightning as L
from datasets import dataset_extr_to_mano
import glob
import json
import numpy as np
from torch.utils.data import DataLoader
import os
import random
import torch

class MyDatamodule(L.LightningDataModule):
    def __init__(self, data_path, bg_rm_dilation, use_grabcut):
        super().__init__()
        self.batch_size = 1
        self.data_path = data_path
        self.bg_rm_dilation = bg_rm_dilation
        self.use_grabcut = use_grabcut

    def setup(self, stage=None):
        # all_ids = list(range(len(
        #     glob.glob(os.path.join(self.data_path, 'images', '*.png'))
        # )))

        all_files = glob.glob(os.path.join(self.data_path, 'images', '*.png'))
        all_ids = sorted([int(os.path.basename(f)[:-4]) for f in all_files])

        # use 80% of the data for training, randomize the order
        random.Random(4).shuffle(all_ids)

        train_ids = all_ids[int(0.2 * len(all_ids)):]
        test_ids = all_ids[:int(0.2 * len(all_ids))]

        # test_and_some_train_ids = all_ids[int(0.4 * len(all_ids)):]

        n_ids_per_split = 50 * torch.cuda.device_count()
        test_and_some_train_ids = test_ids[:n_ids_per_split] + train_ids[:n_ids_per_split]


        self.train_dataset = dataset_extr_to_mano.NeumanDataset(
            self.data_path, train_ids, self.bg_rm_dilation, self.use_grabcut
        )
        self.val_dataset = dataset_extr_to_mano.NeumanDataset(
            self.data_path, test_and_some_train_ids, self.bg_rm_dilation, self.use_grabcut
        )
        # self.full_dataset = dataset_extr_to_mano.NeumanDataset(self.data_path, all_ids, self.bg_rm_dilation)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

