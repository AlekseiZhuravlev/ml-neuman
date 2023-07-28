import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
)
from pytorch3d.implicitron.dataset.rendered_mesh_dataset_map_provider import RenderedMeshDatasetMapProvider
import numpy as np
import torchvision

import sys
sys.path.append("/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf")

from mano_custom import mano_pytorch3d
from render_utils.render_mesh import render_mesh
from render_utils.render_point_cloud import render_point_cloud
from losses.canonical_utils.load_trained_nerf import load_small_nerf, get_random_batch

def get_cow_R_T():
    cow_provider = RenderedMeshDatasetMapProvider(
        data_file="data/cow_mesh/cow.obj",
        use_point_light=False,
        resolution=80,
    )
    dataset_map = cow_provider.get_dataset_map()
    tr_cameras = [training_frame.camera for training_frame in dataset_map.train]

    Rs = torch.stack([tr_camera.R[0] for tr_camera in tr_cameras])
    Ts = torch.stack([tr_camera.T[0] for tr_camera in tr_cameras])
    Ts[:, 2] = 0.3

    return Rs, Ts

def get_look_at_view_R_T(n_cameras):
    # get trajectory rotating around 0,0,0
    Rs = torch.tensor([])
    Ts = torch.tensor([])

    print(360 * torch.rand(1)[0])

    for i in range(n_cameras):
        R, T = look_at_view_transform(
            dist=0.3,
            # elev = 360.0 * torch.rand(1)[0],
            # elev=0.0,
            elev=360.0 / n_cameras * i,
            azim=360.0 / n_cameras * i,
            # azim=0.0,
            # azim=360.0 * torch.rand(1)[0],
        )
        Rs = torch.cat([Rs, R], dim=0)
        Ts = torch.cat([Ts, T], dim=0)
    return Rs, Ts

def get_so3_R_T():
    logRs = torch.zeros(1, 3)
    logRs[:, 1] = torch.linspace(-3.14, 3.14, 1)
    Rs = so3_exp_map(logRs)

    Ts = torch.zeros(1, 3)
    Ts[:, 2] = 0.3
    return Rs, Ts

def create_canonical_cameras(n_cameras, device):
    Rs, Ts = get_look_at_view_R_T(n_cameras)

    batch_cameras = FoVPerspectiveCameras(
        R=Rs,
        T=Ts,
        znear=0.01,
        zfar=10,
        device=device
    )
    return batch_cameras

def render_zero_pose(cameras):
    # n_cameras = 10
    # cameras = create_canonical_cameras(n_cameras, device)

    camera_params, images, silhouettes, manos = get_random_batch()

    hand_model = mano_pytorch3d.create_mano_custom(return_right_hand=False).to(device)

    # verts_py3d = hand_model.get_flat_hand_vertices_pytorch3d(device)

    verts_xyz, _ = hand_model.verts_transformations_xyz(
        betas=manos['shape'].to(device),
        global_orient=manos['root_pose'].to(device),
        hand_pose=manos['hand_pose'].to(device),
        transl=manos['trans'].to(device)
    )
    verts_py3d =  torch.stack((-verts_xyz[:, :, 0], -verts_xyz[:, :, 1], verts_xyz[:, :, 2]),
                            2)


    faces = torch.from_numpy(hand_model.faces.astype(np.int32))[None, :, :].to(device)

    # copy verts_py3d into 1st dimension of batch, number = 10
    verts_py3d = verts_py3d.repeat(n_cameras, 1, 1)
    faces = faces.repeat(n_cameras, 1, 1)

    img, depth = render_mesh(verts_py3d, faces, cameras)
    return img, depth

    # switch n of channels in img
    # img = img.permute(0, 3, 1, 2)
    #
    # # make grid of images and save them
    # grid = torchvision.utils.make_grid(img, nrow=10)
    # torchvision.utils.save_image(grid, "/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/losses/grid.png")
    #
    # # make depth 3 channels
    # depth = depth.repeat(1, 1, 1, 3)
    # depth = depth.permute(0, 3, 1, 2)
    # # make grid of depth maps and save them
    # grid = torchvision.utils.make_grid(depth, nrow=10)
    # torchvision.utils.save_image(grid, "/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/losses/grid_depth.png")

def render_nerf_point_cloud(cameras):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = load_small_nerf()

    # generate points in 3d space between -0.3 and 0.3
    with torch.no_grad():
        point_cloud = torch.linspace(-0.2, 0.2, 200, device='cuda')
        points = torch.stack(torch.meshgrid(point_cloud, point_cloud, point_cloud), dim=-1).reshape(1, 1, -1, 1, 3)
        directions = torch.zeros(1, 1, points.shape[2], 3, device=points.device)
    print('points', points.shape)
    print('directions', directions.shape)

    from tqdm import tqdm

    # iterate over points in chunks of 1000000:
    rays_densities = torch.tensor([], device='cuda')
    rays_colors = torch.tensor([], device='cuda')

    with torch.no_grad():
        for i in tqdm(range(0, points.shape[2], 1000000)):
            points_batch = points[:, :, i:i + 1000000, :, :].to('cuda')
            directions_batch = directions[:, :, i:i + 1000000, :].to('cuda')
            rays_densities_batch, rays_colors_batch = model.get_nerf_output_legacy(points_batch, directions_batch)

            rays_densities = torch.cat((rays_densities, rays_densities_batch), dim=2)
            rays_colors = torch.cat((rays_colors, rays_colors_batch), dim=2)

    points_reshaped = points.reshape(-1, 3)
    rays_colors_reshaped = rays_colors.reshape(-1, 3)
    rays_densities_reshaped = rays_densities.reshape(-1, 1)

    threshold = 0.01

    points_reshaped = points_reshaped[rays_densities.reshape(-1) > threshold]
    rays_colors_reshaped = rays_colors_reshaped[rays_densities.reshape(-1) > threshold]
    rays_densities_reshaped = rays_densities_reshaped[rays_densities.reshape(-1) > threshold]

    rays_colors_rgba = torch.cat((rays_colors_reshaped, rays_densities_reshaped), axis=-1)

    points_reshaped = points_reshaped.reshape(1, -1, 3)
    rays_colors_rgba = rays_colors_rgba.reshape(1, -1, 4)

    # n_cameras = 10
    # cameras = create_canonical_cameras(n_cameras, device)

    points_reshaped = points_reshaped.repeat(n_cameras, 1, 1)
    rays_colors_rgba = rays_colors_rgba.repeat(n_cameras, 1, 1)

    img, depth = render_point_cloud(points_reshaped, rays_colors_rgba, cameras, True, background_color=(1, 1, 1))
    return img, depth


if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    n_cameras = 10
    cameras = create_canonical_cameras(n_cameras, device)

    # print(len(cameras))
    # exit()

    img_zero_pose, depth_zero_pose = render_zero_pose(cameras)
    img_nerf_point_cloud, depth_nerf_point_cloud = render_nerf_point_cloud(cameras)

    target_opacity = depth_zero_pose > 0
    loss = nn.MSELoss(reduction='mean')(depth_nerf_point_cloud, target_opacity)
    print('loss', loss)
    # exit(0)

    print('depth_nerf_point_cloud', depth_nerf_point_cloud)

    # depth_nerf_point_cloud = (depth_nerf_point_cloud > 0).float()

    blended_imgs = torch.tensor([], device='cuda')
    for i in range(10):
        img_nerf = img_nerf_point_cloud[i].clone()
        img_zero = img_zero_pose[i].clone()
        depth_zero = depth_zero_pose[i].clone()

        render_mask = depth_zero > 0

        blended = img_zero * render_mask + img_nerf * (~render_mask)
        blended_imgs = torch.cat((blended_imgs, blended.unsqueeze(0)), dim=0)

    print(img_zero_pose.shape)
    print(img_nerf_point_cloud.shape)
    print(blended_imgs.shape)

    # concatenate image files
    img = torch.cat((img_zero_pose, img_nerf_point_cloud), dim=0)
    depth = torch.cat((depth_zero_pose, depth_nerf_point_cloud), dim=0)

    print(img.shape)
    print(depth.shape)

    # switch n of channels in img
    img = img.permute(0, 3, 1, 2)

    # make grid of images and save them
    grid = torchvision.utils.make_grid(img, nrow=10)
    torchvision.utils.save_image(grid, "/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/losses/canonical_utils/images/pcloud_mesh_grid_img_zero.png")

    # make depth 3 channels
    depth = depth.repeat(1, 1, 1, 3)
    depth = depth.permute(0, 3, 1, 2)
    # make grid of depth maps and save them
    grid = torchvision.utils.make_grid(depth, nrow=10)
    torchvision.utils.save_image(grid, "/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/losses/canonical_utils/images/pcloud_mesh_grid_depth_zero.png")

    # make grid of blended images and save them
    blended_imgs = blended_imgs.permute(0, 3, 1, 2)
    grid = torchvision.utils.make_grid(blended_imgs, nrow=10)
    torchvision.utils.save_image(grid, "/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/losses/canonical_utils/images/pcloud_mesh_grid_blended_zero.png")




