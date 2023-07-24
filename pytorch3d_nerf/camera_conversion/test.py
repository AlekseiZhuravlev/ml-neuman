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


def brute_force_campose(campos, camrot, verts_orig, faces, images, camera_params):

    save_dir = '/itet-stor/azhuavlev/net_scratch/Projects/Results/neuman_custom/camera_brute_force/'
    curr_iter = -1
    for x_ind in [0, 1, 2]:
        # break
        for y_ind in [0, 1, 2]:
            for z_ind in [0, 1, 2]:
                for x_sign in [-1, 1]:
                    for y_sign in [-1, 1]:
                        for z_sign in [-1, 1]:
                            curr_iter += 1

                            R_mat = camrot.clone()
                            t_mat = -np.dot(R_mat, campos.reshape(3, 1)).reshape(3)  # -Rt -> t
                            t_mat = torch.from_numpy(t_mat).float().unsqueeze(0)

                            R_mat = R_mat.clone().squeeze(0).to(device)
                            R_mat = torch.stack((
                                x_sign * R_mat[x_ind],
                                y_sign * R_mat[y_ind],
                                z_sign * R_mat[z_ind]
                            ), 0)
                            R_mat = R_mat.unsqueeze(0)

                            t_mat = t_mat.squeeze(0).to(device)
                            t_mat = torch.stack((
                                x_sign * t_mat[x_ind],
                                y_sign * t_mat[y_ind],
                                z_sign * t_mat[z_ind]
                            ), 0)
                            t_mat = t_mat.unsqueeze(0)

                            verts = verts_orig.clone().squeeze(0).to(device)
                            verts = torch.stack((
                                x_sign * verts[:, x_ind],
                                y_sign * verts[:, y_ind],
                                z_sign * verts[:, z_ind]
                            ), 1)
                            verts = verts.unsqueeze(0)

                            camera = PerspectiveCameras(
                                R=R_mat,
                                T=t_mat,
                                focal_length=camera_params['focal'],
                                principal_point=camera_params['princpt'],
                                in_ndc=False,
                                image_size=camera_params['image_size'],
                                device='cpu'
                            )
                            # %matplotlib inline

                            # from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
                            # camera = cameras_from_opencv_projection(
                            #     R_mat,
                            #     t_mat,
                            #     camera_params['intrinsic_mat'],
                            #     camera_params['image_size'],
                            # )
                            camera = camera.to(device)
                            # verts = torch.stack((verts[:,:,0], verts[:,:,1], verts[:,:,2]),2)

                            img, depth = render_mesh(verts, faces, camera)

                            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

                            ax[0].imshow(img[0].cpu().numpy().astype(np.uint8))
                            ax[1].imshow(images[0].cpu().numpy())

                            fig.savefig(save_dir + f'{x_ind}{y_ind}{z_ind}{x_sign}{y_sign}{z_sign}.png')
                            plt.close(fig)
                            print(curr_iter)


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

    hand_model = mano_pytorch3d.MANOCustom(
        model_path='/home/azhuavlev/Desktop/Data/models/mano/MANO_LEFT.pkl',
        is_rhand=False,
        use_pca=False,
    )

    # verts = example[3]['verts'].cuda()
    camera_params, images, silhouettes, manos = example

    verts_orig = hand_model.forward(
        betas=manos['shape'],
        global_orient=manos['root_pose'],
        hand_pose=manos['hand_pose'],
        transl=manos['trans'],
    ).vertices.to(device)

    faces = torch.from_numpy(hand_model.faces.astype(np.int32))[None, :, :].to(device)

    # R_mat = camera_params['R_pytorch3d']
    # T_mat = camera_params['t_pytorch3d']

    campos = camera_params['campos']
    camrot = camera_params['camrot']

    with torch.no_grad():
        mano_output = hand_model.forward(
            betas=manos['shape'],
            global_orient=manos['root_pose'],
            hand_pose=manos['hand_pose'],
            transl=manos['trans'],
        )
    verts_xyz = mano_output.vertices
    joints_xyz = mano_output.joints

    joints = manos['joints']

    print(torch.tensor(joints['world_coord']).shape)
    print(torch.tensor(joints['joint_valid']).shape)
    # keep only valid joints
    valid_joints = torch.tensor(joints['world_coord'])[torch.tensor(joints['joint_valid']).resize(42).bool()] / 1000
    print(valid_joints.shape)

    # plot valid joints as 3d points
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(valid_joints[:, 0], valid_joints[:, 1], valid_joints[:, 2])
    # ax.scatter(joints_xyz[0, :, 0], joints_xyz[0, :, 1], joints_xyz[0, :, 2])
    # plt.show()
    #
    # exit()

    # verts_xyz = torch.stack((
    #     1 * verts_xyz[:, :, 0],
    #     1 * verts_xyz[:, :, 2],
    #     -1 * verts_xyz[:, :, 1],
    # ), 2)



    cam_coord = np.dot(camrot[0], (verts_xyz[0] - campos[0]).transpose(0, 1)).transpose(1, 0)
    print(cam_coord.shape)

    print(camera_params['focal'], camera_params['princpt'])
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * camera_params['focal'][0][0].numpy() + camera_params['princpt'][0][0].numpy()
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * camera_params['focal'][0][1].numpy() + camera_params['princpt'][0][1].numpy()
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)

    # plot cam coord
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(verts_xyz[0, :, 0], verts_xyz[0, :, 1], verts_xyz[0, :, 2], s=2)
    ax.scatter(campos[0, 0], campos[0, 2], -campos[0, 1], s=10)

    ax = fig.add_subplot(132)
    ax.scatter(img_coord[:, 0], img_coord[:, 1], s=2)
    # ax.set_xlim(0, 400)
    # ax.set_ylim(0, 600)

    print(img_coord[:, 0:2])

    ax = fig.add_subplot(133)
    ax.imshow(images[0])
    ax.scatter(img_coord[:, 0], img_coord[:, 1], s=2)
    print(img_coord[:, 0:2])
    plt.show()
    exit()



    # plot the vertices as 3d points
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(131, projection='3d')

    # plot vertices as matplotlib mesh
    # verts_projected = np.dot(R_mat[0], verts_xyz[0].transpose(1,0)).transpose(1,0) + t_mat.reshape(1,3).numpy()

    # ax.scatter(verts_projected[0, :, 0], verts_projected[0, :, 1], verts_projected[0, :, 2], s=2)

    # verts_xyz *= 1000
    ax.scatter(verts_xyz[0, :, 0], verts_xyz[0, :, 1], verts_xyz[0, :, 2], s=2)

    # campos *= 1000
    print('campos', campos)
    print(manos)
    ax.scatter(campos[0, 0], campos[0, 2], -campos[0, 1], s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax = fig.add_subplot(132)
    # print(images[0].shape)
    ax.imshow(images[0])

    # ax = fig.add_subplot(133)
    # plt.show()
    # exit(0)

    from pytorch3d.transforms.so3 import so3_exponential_map, so3_log_map
    import cv2 as cv

    save_dir = '/itet-stor/azhuavlev/net_scratch/Projects/Results/neuman_custom/camera_brute_force/'
    curr_iter = -1
    for x_ind in [0, 1, 2]:
        # break
        for y_ind in [0, 1, 2]:
            for z_ind in [0, 1, 2]:
                for x_sign in [-1, 1]:
                    for y_sign in [-1, 1]:
                        for z_sign in [-1, 1]:

                            # R_mat = camrot.clone()
                            # t_mat = -np.dot(R_mat, campos.reshape(3, 1)).reshape(3)  # -Rt -> t
                            # t_mat = torch.from_numpy(t_mat).float().unsqueeze(0)
                            #
                            # # convert to opencv format
                            # # R_mat = R_mat.permute(0, 2, 1)
                            # R_mat = torch.linalg.inv(R_mat)
                            #
                            #
                            #
                            # verts_opencv = torch.stack((
                            #     x_sign * verts_xyz[:, :, x_ind],
                            #     y_sign * verts_xyz[:, :, y_ind],
                            #     z_sign * verts_xyz[:, :, z_ind],
                            # ) , 2)
                            #
                            #
                            #
                            # # print(R_mat.shape)
                            # # rot_vec = so3_log_map(R_mat)
                            # # print(R_mat)
                            # rot_vec = cv.Rodrigues(R_mat[0].numpy())[0]
                            # # print(rot_vec)
                            # # exit(0)
                            # imagePoints = \
                            # cv.projectPoints(verts_opencv[0].numpy(), rot_vec, t_mat.numpy(), camera_params['intrinsic_mat'][0].numpy(),
                            #                  None)[0]

                            verts_rearranged = torch.stack((
                                x_sign * verts_xyz[:, :, x_ind],
                                y_sign * verts_xyz[:, :, y_ind],
                                z_sign * verts_xyz[:, :, z_ind],
                            ) , 2)

                            cam_coord = np.dot(camrot[0], (verts_rearranged[0] - campos[0]).transpose(0, 1)).transpose(1, 0)
                            # print(cam_coord.shape)

                            # print(camera_params['focal'], camera_params['princpt'])
                            x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * camera_params['focal'][0][0].numpy() + \
                                camera_params['princpt'][0][0].numpy()
                            y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * camera_params['focal'][0][1].numpy() + \
                                camera_params['princpt'][0][1].numpy()
                            z = cam_coord[:, 2]
                            imagePoints = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
                            imagePoints = imagePoints.reshape(1, 778, 3)



                            # add 2 subplots
                            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                            ax.scatter(imagePoints[0, :, 0], imagePoints[0, :, 1], s=2)
                            ax.imshow(images[0])

                            plt.savefig(os.path.join(save_dir, str(curr_iter) + '.png'))
                            plt.close(fig)
                            # plt.show()
                            curr_iter += 1
                            print(curr_iter)




    # brute_force_campose(campos, camrot, verts_orig, faces, images, camera_params)
