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

from helpers import *

from torch.utils.data import DataLoader
from datasets import dataset_extr_to_mano
import lighning_models
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from nerf_models import nerf_small_no_warp
from nerf_models import nerf_big_no_warp

from lightning.pytorch.strategies import DDPStrategy


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = '/home/azhuavlev/Desktop/Data/InterHand_Neuman/03'

    all_ids = list(range(len(
        glob.glob(os.path.join(data_path, 'images', '*.png'))
    )))

    # use 80% of the data for training, randomize the order
    np.random.shuffle(all_ids)
    train_ids = all_ids[int(0.3 * len(all_ids)):]
    test_ids = all_ids[:int(0.3 * len(all_ids))]
    print(test_ids)

    train_dataset = dataset_extr_to_mano.NeumanDataset(data_path, train_ids)
    test_dataset = dataset_extr_to_mano.NeumanDataset(data_path, test_ids)
    full_dataset = dataset_extr_to_mano.NeumanDataset(data_path, all_ids)

    # We sample 6 random cameras in a minibatch.
    batch_size = 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=5)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=5)

    # initialize nerf model
    train_small = True

    if train_small:
        nerf = nerf_small_no_warp.NeuralRadianceField(input_dim=3)
        offset_net = nerf_small_no_warp.NeuralRadianceField(input_dim=4)
    else:
        raise NotImplementedError

    model = lighning_models.HandModel(nerf_model=nerf, offset_net=offset_net)

    # print(model.raysampler_mc.min_depth)

    output_dir = '/home/azhuavlev/Desktop/Results/neuman_custom/'
    logger = TensorBoardLogger(
        output_dir,
        version='testCanLoss_worldLossesOnly'
    )

    trainer = L.Trainer(
        max_epochs=5001,
        # benchmark=True,
        logger=logger,
        default_root_dir=output_dir,
        check_val_every_n_epoch=250,
        log_every_n_steps=20,
        callbacks=[
            # checkpoint_callback,
            # stats_monitor,
        ],
        # strategy=DDPStrategy(find_unused_parameters=True),
    )
    trainer.fit(
        model,
        train_loader,
        full_loader,
        # ckpt_path='/home/azhuavlev/Desktop/Results/neuman_custom/lightning_logs/version_13/checkpoints/epoch=999-step=48000.ckpt'
    )


