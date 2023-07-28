import sys
sys.path.append("/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf")
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import glob
import os
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

from datasets import dataset_extr_to_mano, dataset_canonical_space
import lighning_models

import nerf_models.nerf_small_both_warp as nerf
from nerf_models import nerf_big_no_warp


def load_small_nerf():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # disable pytorch gradient computation
    torch.set_grad_enabled(False)

    data_path = '/home/azhuavlev/Desktop/Data/InterHand_Neuman/03'
    all_ids = list(range(len(glob.glob(os.path.join(data_path, 'cameras', '*.json')))))

    # use 80% of the data for training, randomize the order
    np.random.shuffle(all_ids)

    full_dataset = dataset_extr_to_mano.NeumanDataset(data_path, all_ids)

    # We sample 6 random cameras in a minibatch.
    batch_size = 1

    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    field = nerf.NeuralRadianceField()

    ckpt_path = '/home/azhuavlev/Desktop/Results/neuman_custom/lightning_logs/small_warp_clipped_sil_loss_99999_lr_99999_mask_0.3_dilation_10_sampling_8192_32_depth_105_huber/checkpoints/epoch=4999-step=180000.ckpt'

    model = lighning_models.HandModel(nerf_model=field).load_from_checkpoint(ckpt_path, nerf_model=field)
    model = model.to('cuda')

    return model

def get_random_batch():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # disable pytorch gradient computation
    torch.set_grad_enabled(False)

    data_path = '/home/azhuavlev/Desktop/Data/InterHand_Neuman/03'
    all_ids = list(range(len(glob.glob(os.path.join(data_path, 'cameras', '*.json')))))

    # use 80% of the data for training, randomize the order
    np.random.shuffle(all_ids)

    full_dataset = dataset_extr_to_mano.NeumanDataset(data_path, all_ids)

    # We sample 6 random cameras in a minibatch.
    batch_size = 1

    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    example = next(iter(full_loader))
    return example


