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
import torch.functional as F
import torch.nn as nn
import torchvision
from IPython import display
from PIL import Image
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras,
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

from mano_custom import mano_pytorch3d
import sampling_utils
from helpers import *
from plot_image_grid import image_grid
from losses import huber
import warp_points


class HandModel(L.LightningModule):
    def __init__(self, dataset, nerf_model):
        super().__init__()

        self.sil_loss_epochs = 999000
        self.sil_loss_start_factor = 0.1

        self.hand_model = mano_pytorch3d.create_mano_custom(return_right_hand=False)

        self.min_depth = 0.1
        self.max_depth = 2

        self.render_size_x = 512
        self.render_size_y = 334

        self.raysampler_train = NDCMultinomialRaysampler(
            image_height=self.render_size_x,
            image_width=self.render_size_y,
            n_rays_per_image=8192,
            n_pts_per_ray=32,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            stratified_sampling=True,

        )
        self.raysampler_test = NDCMultinomialRaysampler(
            image_height=self.render_size_x,
            image_width=self.render_size_y,
            n_pts_per_ray=64,
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
        self.raymarcher = EmissionAbsorptionRaymarcher()

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

        self.loss_func = huber.huber
        # self.loss_func = nn.MSELoss()
        # self.loss_func = nn.L1Loss()


    def configure_optimizers(self):
        # Instantiate the Adam optimizer. We set its master learning rate to 1e-3.
        lr = 5e-4
        optimizer = torch.optim.Adam(self.neural_radiance_field.parameters(), lr=lr)

        # Following the original code, we use exponential decay of the
        # learning rate: current_lr = base_lr * gamma ** (epoch / step_size)
        def lr_lambda(epoch):
            lr_scheduler_gamma = 0.1
            lr_scheduler_step_size = 999000
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

        batch_cameras = PerspectiveCameras(
            R=camera_params['R_pytorch3d'],
            T=camera_params['t_pytorch3d'],
            focal_length=camera_params['focal'],
            principal_point=camera_params['princpt'],
            in_ndc=False,
            image_size=camera_params['image_size'],
            device=self.device
        )

        depths = batch_cameras.get_world_to_view_transform().transform_points(
            manos['verts']
        )[:, :, 2:]

        min_depth = depths.min() * 0.95
        max_depth = depths.max() * 1.05

        masks_sampling = sampling_utils.make_sampling_mask(
            silhouettes
        )


        ###############################################################
        # Ray sampling + warping
        ###############################################################

        ray_bundle = self.raysampler(
            cameras=batch_cameras,
            mask=masks_sampling,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        # Warp the rays to the canonical view.
        rays_points_can, ray_directions_can = warp_points.warp_points(
            rays_points_world,
            manos['verts'],
            manos['Ts'],
        )

        ###########################################################################
        # Rendering
        ###########################################################################

        # get output of nerf model
        rays_densities, rays_features = self.neural_radiance_field(
            rays_points=rays_points_can, ray_directions=ray_directions_can
        )

        # render the images and silhouettes
        rendered_images_silhouettes = self.raymarcher(
            rays_densities=rays_densities,
            rays_features=rays_features,
        )
        rendered_images, rendered_silhouettes = (
            rendered_images_silhouettes.split([3, 1], dim=-1)
        )
        assert rendered_images.isnan().any() == False
        assert rendered_silhouettes.isnan().any() == False

        print('rendered_images_silhouettes', rendered_images_silhouettes.shape)
        exit(0)

        ###########################################################################
        # Silhouette loss, canonical space
        ###########################################################################

        # TODO write new loss
        # TODO change batched forward
        # TODO change validation step
        # TODO add offset network
        # TODO check neuman for other details to implement

        ###########################################################################
        # Silhouette loss, world space
        ###########################################################################

        silhouettes_at_rays = sample_images_at_mc_locs(
            silhouettes.unsqueeze(-1),
            sampled_rays.xys
        )
        assert silhouettes_at_rays.isnan().any() == False

        sil_err = self.loss_func(
            rendered_silhouettes,
            silhouettes_at_rays,
        )
        assert sil_err.isnan() == False

        self.log('sil_loss_unconstrained', sil_err, prog_bar=True, logger=True)

        # decrease silhouette loss and update the factor
        if self.sil_loss_epochs > 0:
            sil_loss_factor = self.sil_loss_start_factor * max(0, 1 - (self.current_epoch / self.sil_loss_epochs))
        else:
            sil_loss_factor = 0
        sil_err = sil_err * sil_loss_factor

        ###########################################################################
        # Color loss
        ###########################################################################

        colors_at_rays = sample_images_at_mc_locs(
            images,
            sampled_rays.xys
        )
        assert colors_at_rays.isnan().any() == False

        color_err = self.loss_func(
            rendered_images,
            colors_at_rays,
        )
        assert color_err.isnan() == False

        ###########################################################################

        # Log the errors.
        self.log('color_loss', color_err, prog_bar=True, logger=True)
        self.log('sil_loss', sil_err, prog_bar=True, logger=True)

        # The optimization loss is a simple
        # sum of the color and silhouette errors.
        loss = color_err + sil_err
        return loss

    def validation_step(self, batch, batch_idx):
        result = self.visualize_batch(batch, warp_rays=True, cameras_canonical=False, canonical_renderer=False)
        self.validation_images.append(result)

    def on_validation_epoch_end(self):
        grid = torchvision.utils.make_grid(self.validation_images, nrow=5)
        self.validation_images = []

        tensorboard_logger = self.logger.experiment
        tensorboard_logger.add_image(f'model_output_{self.global_rank}', grid, self.current_epoch)

    def test_step(self, batch, batch_idx):
        result = self.visualize_batch(batch, warp_rays=False, cameras_canonical=True, canonical_renderer=True)
        self.test_images.append(result)

    def on_test_epoch_end(self):
        grid = torchvision.utils.make_grid(self.test_images, nrow=5, padding=2, pad_value=1)
        # self.test_images = []

        tensorboard_logger = self.logger.experiment
        tensorboard_logger.add_image(f'model_output_test_{self.global_rank}', grid, self.current_epoch)

    def visualize_batch(self, batch, warp_rays, cameras_canonical, canonical_renderer):

        camera_params, images, silhouettes, manos = batch

        if not cameras_canonical:
            batch_cameras = PerspectiveCameras(
                R=camera_params['R_pytorch3d'],
                T=camera_params['t_pytorch3d'],
                focal_length=camera_params['focal'],
                principal_point=camera_params['princpt'],
                in_ndc=False,
                image_size=camera_params['image_size'],
                device=self.device
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

    def get_nerf_output(self, points, directions):
        ray_densities, ray_colors = self.neural_radiance_field.forward(
            points,
            directions,
        )
        return ray_densities, ray_colors
