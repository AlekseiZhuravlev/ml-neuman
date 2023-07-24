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


from helpers import *
from nerf import *

# from data_io import neuman_helper

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load the data
# Datasets
from torch.utils.data import DataLoader
import dataset_from_files
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
import mano_pytorch3d

def render_mesh(mesh, face, cameras):
    batch_size, vertex_num = mesh.shape[:2]

    textures = TexturesVertex(verts_features=torch.ones((batch_size, vertex_num, 3)).float().to(device))
    mesh = Meshes(mesh, face, textures)

    img_size = (512, 334)
    raster_settings = RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1,
                                            perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)
    lights = PointLights(device=device)
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    materials = Materials(
        device=device,
        specular_color=((0.0, 1.0, 0.0),),
        shininess=0
    )

    # render
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)
        images = images[:, :, :, :3] * 255
        depthmaps = fragments.zbuf

    return images, depthmaps


if __name__ == '__main__':
    data_path = '/home/azhuavlev/Desktop/Data/InterHand_Neuman/05_joints'
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

    # print example batch
    example = next(iter(train_loader))

    # hand_model = mano_pytorch3d.MANOCustom(
    #     model_path='/home/azhuavlev/Desktop/Data/models/mano/MANO_LEFT.pkl',
    #     is_rhand=False,
    #     use_pca=False,
    # )
    hand_model = smplx.MANO('/home/azhuavlev/Desktop/Data/models/mano/MANO_LEFT.pkl', use_pca=False, is_rhand=False, flat_hand_mean=False)
    # if torch.sum(torch.abs(ha['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
    #     print('Fix shapedirs bug of MANO')
    #     mano_layer['left'].shapedirs[:, 0, :] *= -1

    hand_model.shapedirs[:, 0, :] *= -1

    # verts = example[3]['verts'].cuda()

    # for batch in test_loader:
    for i in range(len(test_dataset)):

        # TODO is dataloader the issue?
        camera_params, images, silhouettes, manos = test_dataset[i]

        # unsqueeze each tensor
        for key in camera_params.keys():
            camera_params[key] = camera_params[key][None, :]
        images = images[None, :, :, :]
        for key in manos.keys():
            manos[key] = manos[key][None, :]

        verts_orig = hand_model.forward(
            betas=manos['shape'],
            global_orient=manos['root_pose'],
            hand_pose=manos['hand_pose'],
            transl=manos['trans'],
        ).vertices.to(device)

        faces = torch.from_numpy(hand_model.faces.astype(np.int32))[None, :, :].to(device)

        campos = camera_params['campos'] * 1000
        camrot = camera_params['camrot']

        with torch.no_grad():
            mano_output = hand_model.forward(
                betas=manos['shape'],
                global_orient=manos['root_pose'],
                hand_pose=manos['hand_pose'],
                transl=manos['trans'],
            )
        verts_xyz = mano_output.vertices * 1000
        joints_xyz = mano_output.joints

        # t, R = torch.tensor(campos, dtype=torch.float32).reshape(3), torch.tensor(camrot, dtype=torch.float32).reshape(3, 3)
        # t = -(R @ t.reshape(3, 1)).reshape(3)  # -Rt -> t
        # mesh = (R @ verts_xyz[0].transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
        # mesh = mesh.unsqueeze(0)

        t, R = np.array(campos, dtype=np.float32).reshape(3), np.array(
            camrot, dtype=np.float32).reshape(3, 3)
        t = -np.dot(R, t.reshape(3, 1)).reshape(3)  # -Rt -> t
        # mesh = np.dot(R, verts_xyz.squeeze(0).transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)

        mesh = verts_xyz.numpy()[0]

        mesh = torch.from_numpy(mesh).unsqueeze(0).to(device) / 1000




        # reverse x- and y-axis following PyTorch3D axis direction
        mesh = torch.stack((-mesh[:, :, 0], -mesh[:, :, 1], mesh[:, :, 2]), 2)

        cameras = PerspectiveCameras(focal_length=camera_params['focal'],
                                    principal_point=camera_params['princpt'],
                                    device=device,
                                    in_ndc=False,
                                    image_size=camera_params['image_size']
                                     )

        img, depth = render_mesh(mesh, faces, cameras)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(img[0].cpu().numpy().astype(np.uint8))
        ax[1].imshow(images[0].cpu().numpy())

        # save figure
        # plt.savefig('test.png')
        plt.show()

        break