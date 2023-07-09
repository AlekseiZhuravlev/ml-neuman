import glob
import json
import lightning as L
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch3d
import sys
import time
import torch
import torch
import torch.nn as nn
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
from helpers import *
from nerf import *
from plot_image_grid import image_grid
import torchvision

import mano_pytorch3d
import sampling_utils

from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection


class HandModel(L.LightningModule):
    def __init__(self, dataset, nerf_model):
        super().__init__()

        self.hand_model = mano_pytorch3d.MANOCustom(
            model_path='/home/azhuavlev/Desktop/Data/models/mano/MANO_LEFT.pkl',
            is_rhand=False,
            use_pca=False,
        )

        self.calculate_camera_parameters(dataset)

        # Here, NDCMultinomialRaysampler generates a rectangular image
        # grid of rays whose coordinates follow the PyTorch3D
        # coordinate conventions.
        self.raysampler_train = NDCMultinomialRaysampler(
            image_height=self.render_size_x,
            image_width=self.render_size_y,
            n_rays_per_image=2048,
            n_pts_per_ray=128,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            stratified_sampling=True,
        )

        self.raysampler_test = NDCMultinomialRaysampler(
            image_height=self.render_size_x,
            image_width=self.render_size_y,
            n_pts_per_ray=32,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            stratified_sampling=False,
        )

        self.raysampler_canonical = NDCMultinomialRaysampler(
            image_height=self.render_size_x,
            image_width=self.render_size_y,
            n_pts_per_ray=96,
            min_depth=0.1,
            max_depth=1,
            stratified_sampling=False,
        )

        # 2) Instantiate the raymarcher.
        # Here, we use the standard EmissionAbsorptionRaymarcher
        # which marches along each ray in order to render
        # the ray into a single 3D color vector
        # and an opacity scalar.
        self.raymarcher = EmissionAbsorptionRaymarcher()

        # Finally, instantiate the implicit renders
        # for both raysamplers.
        self.renderer_train = ImplicitRenderer(
            raysampler=self.raysampler_train, raymarcher=self.raymarcher,
        )
        self.renderer_test = ImplicitRenderer(
            raysampler=self.raysampler_test, raymarcher=self.raymarcher,
        )
        self.renderer_canonical = ImplicitRenderer(
            raysampler=self.raysampler_canonical, raymarcher=self.raymarcher,
        )


        # Instantiate the radiance field model.
        self.neural_radiance_field = nerf_model

        self.validation_images = []
        self.test_images = []


    def calculate_camera_parameters(self, dataset):
        nears = []
        fars = []
        for element in dataset:
            camera_params, images, silhouettes, manos = element

            verts = self.hand_model.forward_pytorch3d(
                betas= manos['shape'],
                global_orient = manos['root_pose'],
                hand_pose = manos['hand_pose'],
                transl=manos['trans'],
            )

            camera = cameras_from_opencv_projection(
                element[0]['R'],
                element[0]['t'],
                element[0]['intrinsic_mat'],
                element[0]['image_size'],
            )
            # calculate min and max depth from camera to hand
            depths = camera.get_world_to_view_transform().transform_points(verts)[:, :, 2:]

            nears.append(depths.min())
            fars.append(depths.max())

        self.min_depth = min(
            min(nears),
            min(fars),
        ) #* 0.8
        self.max_depth = max(
            max(nears),
            max(fars),
        ) #* 1.2

        self.render_size_x = images.shape[1] #// 2
        self.render_size_y = images.shape[2] #// 2

        print('min_depth', self.min_depth)
        print('max_depth', self.max_depth)
        print('render_size_x', self.render_size_x)
        print('render_size_y', self.render_size_y)


    def configure_optimizers(self):
        # TODO lr decay
        # Instantiate the Adam optimizer. We set its master learning rate to 1e-3.
        lr = 5e-4
        optimizer = torch.optim.Adam(self.neural_radiance_field.parameters(), lr=lr)

        # Following the original code, we use exponential decay of the
        # learning rate: current_lr = base_lr * gamma ** (epoch / step_size)
        def lr_lambda(epoch):
            lr_scheduler_gamma = 0.1
            lr_scheduler_step_size = 1500
            return lr_scheduler_gamma ** (
                    epoch / lr_scheduler_step_size
            )
        # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, verbose=False
        )
        return [optimizer], [lr_scheduler]


    def training_step(self, batch, batch_idx):

        camera_params, images, silhouettes, manos = batch

        batch_cameras = cameras_from_opencv_projection(
            camera_params['R'],
            camera_params['t'],
            camera_params['intrinsic_mat'],
            camera_params['image_size'],
        )

        masks_sampling = sampling_utils.make_sampling_mask(
            silhouettes
        )

        # Evaluate the nerf model.
        rendered_images_silhouettes, sampled_rays = self.renderer_train(
            cameras=batch_cameras,
            volumetric_function=self.neural_radiance_field,
            vertices=manos['verts'],
            Ts=manos['Ts'],
            mask=masks_sampling,
            warp_rays=True
        )

        rendered_images, rendered_silhouettes = (
            rendered_images_silhouettes.split([3, 1], dim=-1)
        )

        # Compute the silhouette error as the mean huber
        # loss between the predicted masks and the
        # sampled target silhouettes.
        silhouettes_at_rays = sample_images_at_mc_locs(
            #target_silhouettes[batch_idx, ..., None],
            silhouettes.unsqueeze(-1),
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
            # target_images[batch_idx],
            images,
            sampled_rays.xys
        )
        color_err = huber(
            rendered_images,
            colors_at_rays,
        ).abs().mean()

        # Log the errors.
        self.log('color_loss', color_err, prog_bar=True, logger=True)
        self.log('sil_loss', sil_err, prog_bar=True, logger=True)

        # The optimization loss is a simple
        # sum of the color and silhouette errors.
        loss = color_err + sil_err
        return loss

    def validation_step(self, batch, batch_idx):
        result = self.visualize_batch(batch, warp_rays=True, cameras_opencv=True, canonical_renderer=False)
        self.validation_images.append(result)

    def on_validation_epoch_end(self):
        grid = torchvision.utils.make_grid(self.validation_images, nrow=5)
        self.validation_images = []

        tensorboard_logger = self.logger.experiment
        tensorboard_logger.add_image('model_output', grid, self.current_epoch)

    def test_step(self, batch, batch_idx):
        result = self.visualize_batch(batch, warp_rays=False, cameras_opencv=False, canonical_renderer=True)
        self.test_images.append(result)

    def on_test_epoch_end(self):
        grid = torchvision.utils.make_grid(self.test_images, nrow=5, padding=2, pad_value=1)
        # self.test_images = []

        tensorboard_logger = self.logger.experiment
        tensorboard_logger.add_image('model_output_test', grid, self.current_epoch)

    def visualize_batch(self, batch, warp_rays, cameras_opencv, canonical_renderer):

        camera_params, images, silhouettes, manos = batch

        if cameras_opencv:
            batch_cameras = cameras_from_opencv_projection(
                camera_params['R'],
                camera_params['t'],
                camera_params['intrinsic_mat'],
                camera_params['image_size'],
            )
        else:
            batch_cameras = FoVPerspectiveCameras(
                R=camera_params['R'],
                T=camera_params['t'],
                znear=camera_params['znear'],
                zfar=camera_params['zfar'],
                device=self.device
            )

        # Render using the grid renderer and the
        # batched_forward function of neural_radiance_field.
        if canonical_renderer:
            rendered_image_silhouette, _ = self.renderer_canonical(
                cameras=batch_cameras,
                volumetric_function=self.neural_radiance_field.batched_forward,
                vertices=manos['verts'],
                Ts=manos['Ts'],
                warp_rays=warp_rays
            )
        else:
            rendered_image_silhouette, _ = self.renderer_test(
                cameras=batch_cameras,
                volumetric_function=self.neural_radiance_field.batched_forward,
                vertices=manos['verts'],
                Ts=manos['Ts'],
                warp_rays=warp_rays
            )
        # Split the rendering result to a silhouette render
        # and the image render.
        rendered_image, rendered_silhouette = (
            rendered_image_silhouette[0].split([3, 1], dim=-1)
        )

        # convert rendered silhouette to rgb
        rendered_silhouette = torch.cat([rendered_silhouette, rendered_silhouette, rendered_silhouette], dim=-1)

        # concatenate the rendered image and the silhouette.
        concat_rendered = torch.cat([rendered_image, rendered_silhouette], dim=1)

        # transpose to (C, H, W)
        concat_rendered = concat_rendered.permute(2, 0, 1)

        return concat_rendered

    def get_nerf_output(self, camera):
        ray_bundle = self.raysampler_canonical.forward(
            camera,
            min_depth = 0.1,
            max_depth = 1,
            n_pts_per_ray = 96,
            stratified_sampling = False,
        )
        ray_densities, ray_colors = self.neural_radiance_field.batched_forward(
            ray_bundle,
            vertices=None,
            Ts=None,
            warp_rays=False,
            n_batches=12
        )
        return ray_densities, ray_colors


