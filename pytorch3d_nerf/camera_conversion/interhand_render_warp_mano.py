import cv2
import json
import numpy as np
import os
import os.path as osp
from glob import glob

os.environ["PYOPENGL_PLATFORM"] = "egl"
import smplx
import torch
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
import matplotlib.pyplot as plt
from pytorch3d.ops.perspective_n_points import efficient_pnp
from test_interhand_render_pnp import render_point_cloud

import sys
sys.path.append("/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf")
from mano_custom import mano_pytorch3d

# mano_layer = {'right': smplx.create('/home/azhuavlev/Desktop/Data/models/mano/MANO_RIGHT.pkl',
#                                     'mano', use_pca=False, is_rhand=True, flat_hand_mean=False),
#               'left': smplx.MANO('/home/azhuavlev/Desktop/Data/models/mano/MANO_LEFT.pkl',
#                                  use_pca=False, is_rhand=False, flat_hand_mean=False)}
mano_layer = {
    'right': mano_pytorch3d.create_mano_custom(return_right_hand=True),
    'left': mano_pytorch3d.create_mano_custom(return_right_hand=False)
}

# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
    print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:, 0, :] *= -1

root_path = '/home/azhuavlev/Desktop/Data/InterHand_Neuman/06_clean/'

save_path = '/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/camera_conversion'

joint_regressor = np.load('/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/camera_conversion/J_regressor_mano_ih26m.npy')

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

for i in range(10):
    hand_type = 'left'

    with open(f'/home/azhuavlev/Desktop/Data/InterHand_Neuman/03/mano/{i:05d}.json') as f:
        mano_param = json.load(f)['left']
    with open(f'/home/azhuavlev/Desktop/Data/InterHand_Neuman/03/cameras/{i:05d}.json') as f:
        cam_param = json.load(f)

    img = cv2.imread(f'/home/azhuavlev/Desktop/Data/InterHand_Neuman/03/images/{i:05d}.png')
    img_height, img_width, _ = img.shape

    # get camera extrinsics
    t, R = np.array(cam_param['campos'], dtype=np.float32).reshape(3), np.array(
        cam_param['camrot'], dtype=np.float32).reshape(3, 3)
    t = -np.dot(R, t.reshape(3, 1)).reshape(3)  # -Rt -> t

    # Second world to camera coordinate conversion method: apply camera extrinsics to MANO parameters
    # get MANO 3D mesh coordinates (world coordinate)
    mano_pose = torch.FloatTensor(mano_param['pose']).view(-1, 3)
    root_pose = mano_pose[0].numpy()
    root_pose, _ = cv2.Rodrigues(root_pose)
    root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))  # multiply camera rotation to MANO root pose
    root_pose = torch.from_numpy(root_pose).view(1, 3)
    hand_pose = mano_pose[1:, :].view(1, -1)
    shape = torch.FloatTensor(mano_param['shape']).view(1, -1)

    with torch.no_grad():
        output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose,
                                   betas=shape)  # this is rotation-aligned, but not translation-aligned with the camera coordinates
    mesh = output.vertices[0].detach().numpy()
    joint_from_mesh = np.dot(joint_regressor, mesh)

    # root_j_reg = joint_from_mesh[20, :]
    # root_j_out = output.joints[0, 0, :].numpy()
    # print('root joint - regressor', root_j_reg, 'output', root_j_out, 'diff', np.linalg.norm(root_j_reg - root_j_out))
    # continue

    # compenstate rotation (translation from origin to root joint was not cancled)
    root_joint_idx = 20
    root_joint = joint_from_mesh[root_joint_idx, None, :]
    trans = np.array(mano_param['trans'])
    trans = np.dot(R, trans.reshape(3, 1)).reshape(1, 3) - root_joint + np.dot(R, root_joint.transpose(1, 0)).transpose(
        1, 0) + t / 1000  # change translation vector
    trans = torch.from_numpy(trans).view(1, 3)


    # get MANO vertices
    output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
    mesh = output.vertices[0].numpy() * 1000  # meter to milimeter

    # render
    mesh = torch.from_numpy(mesh).float().to(device)[None, :, :]
    face = torch.from_numpy(mano_layer[hand_type].faces.astype(np.int32)).to(device)[None, :, :]
    focal = torch.FloatTensor(cam_param['focal']).to(device)[None, :]
    princpt = torch.FloatTensor(cam_param['princpt']).to(device)[None, :]

    with torch.no_grad():
        # render_rgb, render_depth = render_mesh(mesh, face, {'focal': focal, 'princpt': princpt},
        #                                        (img_height, img_width), hand_type)
        render_rgb, render_depth = render_point_cloud(mesh, face, {'focal': focal, 'princpt': princpt},
                                                        (img_height, img_width), hand_type)
        # render_rgb, render_depth = project_points(mesh, face, {'focal': focal, 'princpt': princpt},
        #                                                 (img_height, img_width), hand_type, img, mesh_unprojected)

    # blend
    render_rgb = render_rgb[0].cpu().numpy()
    render_depth = render_depth[0].cpu().numpy()

    valid_mask = render_depth > 0
    render_mask = valid_mask

    render_out = render_rgb * render_mask + img * (1 - render_mask)

    cv2.imwrite(osp.join(save_path, f'extr_to_mano/{i:05d}.png'), render_out)
    # plt.imshow(render_out[..., ::-1].astype(np.uint8))
    # plt.show()