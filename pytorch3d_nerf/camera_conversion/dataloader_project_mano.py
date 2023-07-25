import sys

sys.path.append("/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf")

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
import cv2
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    PointLights,
    DirectionalLights,
    PerspectiveCameras,
    Materials,
    SoftPhongShader,
    RasterizationSettings,
    MeshRenderer,
    MeshRendererWithFragments,
    MeshRasterizer,
    TexturesVertex,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
    PointsRasterizationSettings,
)


from helpers import *
from nerf import *

# from data_io import neuman_helper

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load the data
# Datasets
from torch.utils.data import DataLoader
from datasets import dataset_extr_to_mano
import glob

import smplx
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
PointLights,
DirectionalLights,
PerspectiveCameras,
Materials,
SoftPhongShader,
RasterizationSettings,
MeshRenderer,
MeshRendererWithFragments,
MeshRasterizer,
TexturesVertex)
from pytorch3d.transforms.so3 import so3_exponential_map, so3_log_map
import matplotlib.pyplot as plt
from mano_custom import mano_pytorch3d
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from test_interhand_render_pnp import render_mesh, render_point_cloud


if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    data_path = '/home/azhuavlev/Desktop/Data/InterHand_Neuman/03'
    save_path = '/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/camera_conversion'

    all_ids = list(range(len(glob.glob(os.path.join(data_path, 'cameras', '*.json')))))

    # use 80% of the data for training, randomize the order
    np.random.shuffle(all_ids)
    train_ids = all_ids[:int(0.6 * len(all_ids))]
    test_ids = all_ids[int(0.6 * len(all_ids)):]
    print(test_ids)

    train_dataset = dataset_extr_to_mano.NeumanDataset(data_path, train_ids)
    test_dataset = dataset_extr_to_mano.NeumanDataset(data_path, test_ids)
    full_dataset = dataset_extr_to_mano.NeumanDataset(data_path, all_ids)

    # We sample 6 random cameras in a minibatch.
    batch_size = 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    hand_model = mano_pytorch3d.create_mano_custom(return_right_hand=False)

    max_iter = 100
    for i, batch in enumerate(full_loader):
        if i > max_iter:
            break

        camera_params, images, silhouettes, manos = batch

        with torch.no_grad():
            mano_output = hand_model.forward(
                betas=manos['shape'],
                global_orient=manos['root_pose'],
                hand_pose=manos['hand_pose'],
                transl=manos['trans'],
            )
        mesh = mano_output.vertices.to(device)

        # print(mesh)
        # exit()

        face = torch.from_numpy(hand_model.faces.astype(np.int32)).to(device)[None, :, :]

        focal = camera_params['focal'].to(device)
        princpt = camera_params['princpt'].to(device)

        with torch.no_grad():
            render_rgb, render_depth = render_mesh(mesh, face, {'focal': focal, 'princpt': princpt},
                                                          (512, 334), 'left')
        # blend
        render_rgb = render_rgb[0].cpu().numpy()
        render_depth = render_depth[0].cpu().numpy()

        valid_mask = render_depth > 0
        render_mask = valid_mask

        render_out = render_rgb * render_mask + images[0].numpy() * (1 - render_mask)

        # draw 2 plots, render out and images[0]
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(render_out)
        axs[1].imshow(images[0].numpy())

        # plt.show()
        # exit()
        plt.savefig(os.path.join(save_path, f'dataloader_extr_to_mano/{i:05d}.png'))
        plt.close()
        # cv2.imwrite(os.path.join(save_path, f'dataloader_extr_to_mano/{i:05d}.png'), render_out)

        # TODO run script for all images, check that render correctly
        # warp points to canonical space, check quality
        # check warping script again
        # in dataset, replace R, t with ones and zeros, rename original to sth else
        # run small nerf with warping
        # adapt big nerf


