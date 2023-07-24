# Copyright (c) Facebook, Inc. and its affiliates.

import os
import numpy as np
import cv2
import json
from glob import glob
import os.path as osp

os.environ["PYOPENGL_PLATFORM"] = "egl"
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


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0] + 1) + '/' + str(f[i][0] + 1) + ' ' + str(f[i][1] + 1) + '/' + str(
            f[i][1] + 1) + ' ' + str(f[i][2] + 1) + '/' + str(f[i][2] + 1) + '\n')
    obj_file.close()


def render_mesh(mesh, face, cam_param, render_shape, hand_type):
    batch_size, vertex_num = mesh.shape[:2]
    mesh = mesh / 1000  # milimeter to meter

    textures = TexturesVertex(verts_features=torch.ones((batch_size, vertex_num, 3)).float().to(device))
    mesh = torch.stack((-mesh[:, :, 0], -mesh[:, :, 1], mesh[:, :, 2]),
                       2)  # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face, textures)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'],
                                 principal_point=cam_param['princpt'],
                                 device=device,
                                 in_ndc=False,
                                 image_size=torch.LongTensor(render_shape).to(device).view(1, 2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1,
                                            perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)
    lights = PointLights(device=device)
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    if hand_type == 'right':
        color = ((1.0, 0.0, 0.0),)
    else:
        color = ((0.0, 1.0, 0.0),)
    materials = Materials(
        device=device,
        specular_color=color,
        shininess=0
    )

    # render
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)
        images = images[:, :, :, :3] * 255
        depthmaps = fragments.zbuf

    return images, depthmaps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# mano layer
smplx_path = 'SMPLX_PATH'
mano_layer = {'right': smplx.create('/home/azhuavlev/Desktop/Data/models/mano/MANO_RIGHT.pkl',
                                    'mano', use_pca=False, is_rhand=True, flat_hand_mean=False),
              'left': smplx.MANO('/home/azhuavlev/Desktop/Data/models/mano/MANO_LEFT.pkl',
                                 use_pca=False, is_rhand=False, flat_hand_mean=False)}

# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
    print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:, 0, :] *= -1

root_path = '/home/azhuavlev/Desktop/Data/InterHand_Neuman/06_clean/'

save_path = '/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/camera_conversion'

for i in range(10):
    hand_type = 'left'

    with open(f'/home/azhuavlev/Desktop/Data/InterHand_Neuman/03/mano/{i:05d}.json') as f:
        mano_param = json.load(f)['left']
    with open(f'/home/azhuavlev/Desktop/Data/InterHand_Neuman/03/cameras/{i:05d}.json') as f:
        cam_param = json.load(f)

    img = cv2.imread(f'/home/azhuavlev/Desktop/Data/InterHand_Neuman/03/images/{i:05d}.png')
    img_height, img_width, _ = img.shape

    # get MANO 3D mesh coordinates (world coordinate)
    mano_pose = torch.FloatTensor(mano_param['pose']).view(-1, 3)
    root_pose = mano_pose[0].view(1, 3)
    hand_pose = mano_pose[1:, :].view(1, -1)
    shape = torch.FloatTensor(mano_param['shape']).view(1, -1)
    trans = torch.FloatTensor(mano_param['trans']).view(1, 3)
    output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
    mesh = output.vertices[0].numpy() * 1000  # meter to milimeter

    # apply camera extrinsics
    t, R = np.array(cam_param['campos'], dtype=np.float32).reshape(3), np.array(
        cam_param['camrot'], dtype=np.float32).reshape(3, 3)
    t = -np.dot(R, t.reshape(3, 1)).reshape(3)  # -Rt -> t
    mesh = np.dot(R, mesh.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)

    # render
    mesh = torch.from_numpy(mesh).float().to(device)[None, :, :]
    face = torch.from_numpy(mano_layer[hand_type].faces.astype(np.int32)).to(device)[None, :, :]
    focal = torch.FloatTensor(cam_param['focal']).to(device)[None, :]
    princpt = torch.FloatTensor(cam_param['princpt']).to(device)[None, :]
    with torch.no_grad():
        render_rgb, render_depth = render_mesh(mesh, face, {'focal': focal, 'princpt': princpt},
                                               (img_height, img_width), hand_type)

    # blend
    render_rgb = render_rgb[0].cpu().numpy()
    render_depth = render_depth[0].cpu().numpy()

    valid_mask = render_depth > 0
    render_mask = valid_mask

    render_out = render_rgb * render_mask + img * (1 - render_mask)

    print(render_out)

    # save image
    cv2.imwrite(osp.join(save_path, f'{i:05d}.png'), render_out)
    import matplotlib.pyplot as plt
    plt.imshow(render_out.astype(np.uint8))
    plt.show()
    # break