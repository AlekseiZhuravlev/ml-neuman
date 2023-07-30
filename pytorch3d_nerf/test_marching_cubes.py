import sys
sys.path.append("/home/azhuavlev/PycharmProjects/ml-neuman_mano")

import lighning_models
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import nerf
import dataset_canonical_space
from torch.utils.data import DataLoader
import dataset_from_files
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

from pytorch3d_nerf.notebooks.generate_cow_renders import generate_cow_renders
from plot_image_grid import image_grid

from helpers import *
from nerf import *

from data_io import neuman_helper

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    # disable pytorch gradient computation
    torch.set_grad_enabled(False)


    data_path = '/home/azhuavlev/Desktop/Data/InterHand_Neuman/03'
    all_ids = list(range(len(glob.glob(os.path.join(data_path, 'cameras', '*.json')))))

    # use 80% of the data for training, randomize the order
    np.random.shuffle(all_ids)
    train_ids = all_ids[:int(0.6 * len(all_ids))]
    test_ids = all_ids[int(0.6 * len(all_ids)):]
    print(test_ids)

    train_dataset = dataset_from_files.NeumanDataset(data_path, train_ids)
    test_dataset = dataset_from_files.NeumanDataset(data_path, test_ids)
    full_dataset = dataset_from_files.NeumanDataset(data_path, all_ids)

    # We sample 6 random cameras in a minibatch.
    batch_size = 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    nerf = nerf.NeuralRadianceField()

    ckpt_path = '/home/azhuavlev/Desktop/Results/neuman_custom/lightning_logs/version_128/checkpoints/epoch=649-step=23400.ckpt'
    model = lighning_models.HandModel(dataset=full_loader, nerf_model=nerf).load_from_checkpoint(ckpt_path,
                                                                                                 dataset=full_loader,
                                                                                                 nerf_model=nerf)

    can_dataset = dataset_canonical_space.CanSpaceDataset(n_cameras=20)
    can_dataloader = DataLoader(can_dataset, batch_size=1, shuffle=False, num_workers=5)

    trainer = L.Trainer()

    camera_params, images, silhouettes, manos = can_dataset[0:1]

    batch_cameras = FoVPerspectiveCameras(
        R=camera_params['R'],
        T=camera_params['t'],
        znear=camera_params['znear'],
        zfar=camera_params['zfar'],
        device=torch.device("cuda"),
    )
    model = model.to('cuda')

    model.get_nerf_output(
        camera=batch_cameras,
    )