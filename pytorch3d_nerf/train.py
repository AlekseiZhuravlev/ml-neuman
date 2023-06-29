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

from generate_cow_renders import generate_cow_renders
from plot_image_grid import image_grid

from helpers import *
from nerf import *

from data_io import neuman_helper

# obtain the utilized device
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    train_scene = neuman_helper.NeuManReader.read_scene(
        '/home/azhuavlev/Desktop/Data/InterHand_Neuman/01/',
        tgt_size=None,
        normalize=False,
        bkg_range_scale=3,
        human_range_scale=1.5,
        mask_dir='segmentations',
        smpl_type='optimized'
    )

    train_scene.read_data_to_ram(data_list=['image', 'depth'])

    Rs = []
    Ts = []
    target_images, target_silhouettes = [], []
    for i in range(len(train_scene.captures)):

        # print(train_scene.captures[0].pinhole_cam.intrinsic_matrix,
        #       train_scene.captures[0].cam_pose.translation_matrix,
        #       train_scene.captures[0].cam_pose.translation_vector,
        #       train_scene.captures[0].cam_pose.rotation_matrix
        #       )
        Rs.append(train_scene.captures[i].cam_pose.rotation_matrix[0:3, 0:3])
        Ts.append(train_scene.captures[i].cam_pose.translation_vector)

        # not normalized!
        target_images.append(train_scene.captures[i].image)
        target_silhouettes.append(train_scene.captures[i].mask)

    R = torch.tensor(np.array(Rs))
    T = torch.tensor(np.array(Ts))
    target_images = torch.tensor(np.array(target_images))
    target_silhouettes = torch.tensor(np.array(target_silhouettes))

    target_images = (target_images.to(torch.float32) / 255.0).clamp(0.0, 1.0)
    target_silhouettes = target_silhouettes.to(torch.float32)

    print('R', R.shape)
    print('T', T.shape)
    print('target_images', target_images.shape, target_images.dtype, target_images)
    print('target_silhouettes', target_silhouettes.shape, target_silhouettes.dtype, target_silhouettes)

    # Our rendered scene is centered around (0,0,0)
    # and is enclosed inside a bounding box
    # whose side is roughly equal to 3.0 (world units).
    min_depth = 0.01
    volume_extent_world = 2.0

    target_cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=min_depth, zfar=volume_extent_world)

    # target_cameras, target_images, target_silhouettes = generate_cow_renders(num_views=40, azimuth_range=180)
    print(f'Generated {len(target_images)} images/silhouettes/cameras.')

    # print('target_images', target_images.shape, target_images.dtype, target_images)
    # print('target_silhouettes', target_silhouettes.shape, target_silhouettes.dtype, target_silhouettes)


    # render_size describes the size of both sides of the
    # rendered images in pixels. Since an advantage of
    # Neural Radiance Fields are high quality renders
    # with a significant amount of details, we render
    # the implicit function at double the size of
    # target images.
    render_size = target_images.shape[1]


    # 1) Instantiate the raysamplers.

    # Here, NDCMultinomialRaysampler generates a rectangular image
    # grid of rays whose coordinates follow the PyTorch3D
    # coordinate conventions.
    raysampler_grid = NDCMultinomialRaysampler(
        image_height=render_size,
        image_width=render_size,
        n_pts_per_ray=128,
        min_depth=min_depth,
        max_depth=volume_extent_world,
    )

    # MonteCarloRaysampler generates a random subset
    # of `n_rays_per_image` rays emitted from the image plane.
    raysampler_mc = MonteCarloRaysampler(
        min_x=-1.0,
        max_x=1.0,
        min_y=-1.0,
        max_y=1.0,
        n_rays_per_image=750,
        n_pts_per_ray=128,
        min_depth=min_depth,
        max_depth=volume_extent_world,
    )

    # 2) Instantiate the raymarcher.
    # Here, we use the standard EmissionAbsorptionRaymarcher
    # which marches along each ray in order to render
    # the ray into a single 3D color vector
    # and an opacity scalar.
    raymarcher = EmissionAbsorptionRaymarcher()

    # Finally, instantiate the implicit renders
    # for both raysamplers.
    renderer_grid = ImplicitRenderer(
        raysampler=raysampler_grid, raymarcher=raymarcher,
    )
    renderer_mc = ImplicitRenderer(
        raysampler=raysampler_mc, raymarcher=raymarcher,
    )

    # First move all relevant variables to the correct device.
    renderer_grid = renderer_grid.to(device)
    renderer_mc = renderer_mc.to(device)
    target_cameras = target_cameras.to(device)
    target_images = target_images.to(device)
    target_silhouettes = target_silhouettes.to(device)

    # Set the seed for reproducibility
    torch.manual_seed(1)

    # Instantiate the radiance field model.
    neural_radiance_field = NeuralRadianceField().to(device)

    # Instantiate the Adam optimizer. We set its master learning rate to 1e-3.
    lr = 1e-3
    optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=lr)

    # We sample 6 random cameras in a minibatch. Each camera
    # emits raysampler_mc.n_pts_per_image rays.
    batch_size = 1

    # 3000 iterations take ~20 min on a Tesla M40 and lead to
    # reasonably sharp results. However, for the best possible
    # results, we recommend setting n_iter=20000.
    n_iter = 3000

    # Init the loss history buffers.
    loss_history_color, loss_history_sil = [], []

    # The main optimization loop.
    for iteration in tqdm(range(n_iter)):
        # In case we reached the last 75% of iterations,
        # decrease the learning rate of the optimizer 10-fold.
        if iteration == round(n_iter * 0.75):
            print('Decreasing LR 10-fold ...')
            optimizer = torch.optim.Adam(
                neural_radiance_field.parameters(), lr=lr * 0.1
            )

        # Zero the optimizer gradient.
        optimizer.zero_grad()

        # Sample random batch indices.
        batch_idx = torch.randperm(len(target_cameras))[:batch_size]

        # Sample the minibatch of cameras.
        batch_cameras = FoVPerspectiveCameras(
            R=target_cameras.R[batch_idx],
            T=target_cameras.T[batch_idx],
            znear=target_cameras.znear[batch_idx],
            zfar=target_cameras.zfar[batch_idx],
            aspect_ratio=target_cameras.aspect_ratio[batch_idx],
            fov=target_cameras.fov[batch_idx],
            device=device,
        )

        # Evaluate the nerf model.
        rendered_images_silhouettes, sampled_rays = renderer_mc(
            cameras=batch_cameras,
            volumetric_function=neural_radiance_field
        )
        rendered_images, rendered_silhouettes = (
            rendered_images_silhouettes.split([3, 1], dim=-1)
        )

        # Compute the silhouette error as the mean huber
        # loss between the predicted masks and the
        # sampled target silhouettes.
        silhouettes_at_rays = sample_images_at_mc_locs(
            target_silhouettes[batch_idx, ..., None],
            sampled_rays.xys
        )
        sil_err = huber(
            rendered_silhouettes,
            silhouettes_at_rays,
        ).abs().mean()

        # Compute the color error as the mean huber
        # loss between the rendered colors and the
        # sampled target images.
        colors_at_rays = sample_images_at_mc_locs(
            target_images[batch_idx],
            sampled_rays.xys
        )
        color_err = huber(
            rendered_images,
            colors_at_rays,
        ).abs().mean()

        # The optimization loss is a simple
        # sum of the color and silhouette errors.
        loss = color_err + sil_err

        # Log the loss history.
        loss_history_color.append(float(color_err))
        loss_history_sil.append(float(sil_err))

        # Every 10 iterations, print the current values of the losses.
        if iteration % 10 == 0:
            print(
                f'Iteration {iteration:05d}:'
                + f' loss color = {float(color_err):1.2e}'
                + f' loss silhouette = {float(sil_err):1.2e}'
            )

        # Take the optimization step.
        loss.backward()
        optimizer.step()

        # Visualize the full renders every 100 iterations.
        if iteration % 100 == 0:
            show_idx = torch.randperm(len(target_cameras))[:1]
            show_full_render(
                neural_radiance_field,
                FoVPerspectiveCameras(
                    R=target_cameras.R[show_idx],
                    T=target_cameras.T[show_idx],
                    znear=target_cameras.znear[show_idx],
                    zfar=target_cameras.zfar[show_idx],
                    aspect_ratio=target_cameras.aspect_ratio[show_idx],
                    fov=target_cameras.fov[show_idx],
                    device=device,
                ),
                target_images[show_idx][0],
                target_silhouettes[show_idx][0],
                loss_history_color,
                loss_history_sil,
                renderer_grid=renderer_grid,
            )


    with torch.no_grad():
        rotating_nerf_frames = generate_rotating_nerf(neural_radiance_field, n_frames=3 * 5)

    image_grid(rotating_nerf_frames.clamp(0., 1.).cpu().numpy(), rows=3, cols=5, rgb=True, fill=True)
    plt.show()