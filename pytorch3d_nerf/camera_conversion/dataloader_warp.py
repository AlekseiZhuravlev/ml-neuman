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
        verts_xyz = mano_output.vertices.to(device)

        Ts = manos['Ts']
        T_inv = torch.inverse(Ts)

        # add homogeneous coordinate
        verts_xyz_homo = torch.cat([
            verts_xyz,
            torch.ones_like(verts_xyz[..., 0:1])
        ], axis=-1)

        # for each batch, perform torch.bmm. Unsqueeze from [4] to [4, 1] vector
        verts_xyz_homo = verts_xyz_homo.unsqueeze(-1)
        can_verts_xyz = torch.stack(
            [torch.bmm(T_inv[i], verts_xyz_homo[i]) for i in range(T_inv.shape[0])], dim=0)

        # Squeeze from [4, 1] to [4] vector, drop homogeneous dimension
        can_verts_xyz = can_verts_xyz.squeeze(-1)[..., :3]

        # exit()


        # draw 2 plots, render out and images[0]
        fig = plt.figure(figsize=(10, 10))
        axs = fig.add_subplot(1, 2, 1, projection='3d')
        # plot verts_xyz as 3d scatter
        axs.scatter(verts_xyz[0, :, 0], verts_xyz[0, :, 1], verts_xyz[0, :, 2], s=1)
        axs.set_xlabel('X Label')
        axs.set_ylabel('Y Label')
        axs.set_zlabel('Z Label')
        axs.set_title('verts_xyz')

        axs = fig.add_subplot(1, 2, 2, projection='3d')
        # plot verts_xyz as 3d scatter
        axs.scatter(can_verts_xyz[0, :, 0], can_verts_xyz[0, :, 1], can_verts_xyz[0, :, 2], s=1)
        axs.set_xlabel('X Label')
        axs.set_ylabel('Y Label')
        axs.set_zlabel('Z Label')

        axs.axis('equal')
        # set x scale to -0.1 0.1
        #
        # axs.set_xscale(-0.1,0.1)
        # axs.set_yscale(-0.1, 0.1)
        # axs.set_zscale(-0.1, 0.1)
        axs.set_title('verts_warped')

        plt.show()
        # exit()
        plt.savefig(os.path.join(save_path, f'warping/{i:05d}.png'))
        plt.close()
        # cv2.imwrite(os.path.join(save_path, f'dataloader_extr_to_mano/{i:05d}.png'), render_out)

        # TODO run script for all images, check that render correctly
        # warp points to canonical space, check quality
        # check warping script again
        # in dataset, replace R, t with ones and zeros, rename original to sth else
        # run small nerf with warping
        # adapt big nerf


