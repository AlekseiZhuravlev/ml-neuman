import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch3d
import sys
import time
import torch
from IPython import display
from PIL import Image
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)
# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from tqdm import tqdm

import sys
sys.path.append("/home/azhuavlev/PycharmProjects/ml-neuman_mano")

from data_io import neuman_helper
from generate_cow_renders import generate_cow_renders
from helpers import *
from nerf import *
from plot_image_grid import image_grid

from torch.utils.data import DataLoader
import dataset_from_files
import lighning_models
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import nerf_original
import nerf_models.nerf_big_no_warp as nerf_no_warp
import datasets.dataset_single_image as dataset_single_image

from lightning.pytorch.strategies import DDPStrategy

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = '/home/azhuavlev/Desktop/Data/InterHand_Neuman/03'

    all_ids = list(range(len(
        glob.glob(os.path.join(data_path, 'images', '*.png'))
    )))

    # use 80% of the data for training, randomize the order
    np.random.shuffle(all_ids)
    train_ids = all_ids[:int(0.7 * len(all_ids))]
    test_ids = all_ids[int(0.7 * len(all_ids)):]
    # train_ids = all_ids[:10]
    # test_ids = all_ids[10:]
    print(test_ids)

    # We sample 1 random camera in a minibatch.
    batch_size = 1

    # # Use dataset of single image for debugging
    # full_dataset = dataset_single_image.NeumanDataset(data_path, all_ids)
    # train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    # test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=5)
    # full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=5)

    train_dataset = dataset_from_files.NeumanDataset(data_path, train_ids)
    test_dataset = dataset_from_files.NeumanDataset(data_path, test_ids)
    full_dataset = dataset_from_files.NeumanDataset(data_path, all_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=5)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=5)

    # initialize nerf model
    # nerf = nerf_original.NeuralRadianceField()
    nerf = nerf_no_warp.NeuralRadianceField()

    model = lighning_models.HandModel(dataset=full_loader, nerf_model=nerf)

    output_dir = '/home/azhuavlev/Desktop/Results/neuman_custom/'
    logger = TensorBoardLogger(output_dir, version='big_sil_loss_1000_mask_0.05_dilation_50_sampling_8192_32_depth_105_mae_mask_mae')

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(
        max_epochs=2001,
        benchmark=True,
        logger=logger,
        default_root_dir=output_dir,
        check_val_every_n_epoch=200,
        log_every_n_steps=15,
        #TODO changed
        callbacks=[
            lr_monitor,
        ],
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    trainer.fit(
        model,
        train_loader,
        # full_loader,
        # ckpt_path='/home/azhuavlev/Desktop/Results/neuman_custom/lightning_logs/version_13/checkpoints/epoch=999-step=48000.ckpt'
    )


