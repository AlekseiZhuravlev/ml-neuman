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
    nerf = nerf_big_no_warp.NeuralRadianceField()

    model = lighning_models.HandModel(nerf_model=nerf)

    # print(model.raysampler_mc.min_depth)

    output_dir = '/home/azhuavlev/Desktop/Results/neuman_custom/'
    logger = TensorBoardLogger(
        output_dir,
        # version='small_warp_clipped_sil_loss_99999_lr_99999_mask_0.3_dilation_10_sampling_8192_32_depth_105_huber'
        # version='testCanLoss_silFactor_1_canFactor_1_canCam_2_canLoss_huber_silLoss_huber_opacity_-1_noDir'
        # version='testCanLoss_canSampling_lossSilCan_noCopyDirs_disabledManoGrad_disabledOtherLosses_FOVcameras_Stratified_harmonic_0.01'
        version='big_worldLosses'
        # version='testCanLoss_allLosses'
    )
    # checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor="epoch", mode='max', every_n_epochs=1)

    trainer = L.Trainer(
        max_epochs=10001,
        # benchmark=True,
        logger=logger,
        default_root_dir=output_dir,
        check_val_every_n_epoch=500,
        log_every_n_steps=20,
        callbacks=[
            # checkpoint_callback,
            # stats_monitor,
        ],
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    trainer.fit(
        model,
        train_loader,
        full_loader,
        # ckpt_path='/home/azhuavlev/Desktop/Results/neuman_custom/lightning_logs/version_13/checkpoints/epoch=999-step=48000.ckpt'
    )


