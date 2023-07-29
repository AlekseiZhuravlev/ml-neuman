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
from tqdm import tqdm

import sampling_utils
import warp_points
from helpers import sample_images_at_mc_locs
from losses import huber
from losses.opacity_loss import OpacityLoss
from losses.sil_loss_can import SilhouetteLossCanonical
from losses.sil_loss_world import SilhouetteLossWorld
from mano_custom import mano_pytorch3d
from plot_image_grid import image_grid
from losses.canonical_utils.cameras_canonical import create_canonical_cameras


class HandModel(L.LightningModule):
    def __init__(self, nerf_model):
        super().__init__()

        self.hand_model = mano_pytorch3d.create_mano_custom(return_right_hand=False)
        # self.register_buffer(
        #     'faces',
        #     torch.from_numpy(self.hand_model.faces.astype(np.int32))[None, :, :]
        # )
        #
        # # TODO this is not 100% accurate, but it's close enough
        # self.register_buffer(
        #     'verts_zero_pose',
        #     self.hand_model.get_flat_hand_vertices_pytorch3d(self.device)
        # )

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
        self.raysampler_val = NDCMultinomialRaysampler(
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
            n_pts_per_ray=64,
            min_depth=0.01,
            max_depth=1,
            stratified_sampling=False,
        )
        self.raymarcher = EmissionAbsorptionRaymarcher()

        # Instantiate the radiance field model.
        self.neural_radiance_field = nerf_model

        self.validation_images = []
        self.can_validation_images = []

        self.loss_func = huber.huber
        # self.loss_func = nn.MSELoss()
        # self.loss_func = nn.L1Loss()

        self.sil_loss_world = SilhouetteLossWorld(
            loss_func=huber.huber,
            # loss_func=nn.MSELoss(),
            sil_loss_start_factor=1,
            sil_loss_epochs=999000,
        )

        # TODO think about the parameters
        self.opacity_loss = OpacityLoss(
            alpha=3,
            start_factor=10,
            start_epoch=10000,
            max_epochs=999000,
        )

    def on_fit_start(self):

        self.sil_loss_can = SilhouetteLossCanonical(
            n_cameras=5,
            verts_zero_pose=self.hand_model.get_flat_hand_vertices_pytorch3d(self.device),
            faces=torch.from_numpy(self.hand_model.faces.astype(np.int32))[None, :, :].to(self.device),
            loss_func=huber.huber,
            # loss_func=nn.MSELoss(),
            sil_loss_start_factor=1,
            sil_loss_epochs=999000,
            device=self.device,
        )
        self.sil_loss_can.to(self.device)


    def configure_optimizers(self):
        # lr = 5e-4
        lr = 5e-3
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

        ray_bundle = self.raysampler_train(
            cameras=batch_cameras,
            mask=masks_sampling,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        # Warp the rays to the canonical view.
        ray_points_can, ray_directions_can = warp_points.warp_points(
            rays_points_world,
            manos['verts'],
            manos['Ts'],
        )
        assert ray_points_can.isnan().any() == False
        assert ray_directions_can.isnan().any() == False

        ###########################################################################
        # Rendering
        ###########################################################################

        # get output of nerf model
        rays_densities, rays_features = self.neural_radiance_field(
            ray_points=ray_points_can, ray_directions=ray_directions_can
        )
        assert rays_densities.isnan().any() == False
        assert rays_features.isnan().any() == False

        # render the images and silhouettes
        rendered_images_silhouettes = self.raymarcher(
            rays_densities=rays_densities,
            rays_features=rays_features,
        )
        assert rendered_images_silhouettes.isnan().any() == False
        rendered_images, rendered_silhouettes = (
            rendered_images_silhouettes.split([3, 1], dim=-1)
        )
        # rendered_images_silhouettes.shape torch.Size([1, 8192, 1, 4])

        ###########################################################################
        # Silhouette loss, canonical space
        ###########################################################################

        # change mask loss
        # write opacity penalization, either 0 or 1
        # write canonical loss
        # check gradients again
        # add saving of images randomly
        # change batched forward
        # test batched warp
        # TODO change validation step
        # TODO add offset network
        # TODO check neuman for other details to implement

        sil_err_can, sil_err_unconstrained_can, sil_loss_factor_can = self.sil_loss_can.forward(
            ray_points_can,
            rays_densities,
            rays_features,
            self.current_epoch,
        )

        ###########################################################################
        # Silhouette loss, world space
        ###########################################################################

        sil_err_world, sil_err_world_unconstrained, sil_loss_factor_world = self.sil_loss_world.forward(
            rendered_silhouettes,
            silhouettes,
            ray_bundle,
            self.current_epoch,
        )

        # print('sil_err_world', sil_err_world)
        # print('sil_err_world_unconstrained', sil_err_world_unconstrained)
        #
        # print('sil_err_can', sil_err_can)
        # print('sil_err_unconstrained_can', sil_err_unconstrained_can)
        # exit()

        ###########################################################################
        # Opacity loss (either 0 or 1) - needs to be delayed
        ###########################################################################

        opacity_err, opacity_err_unconstrained, opacity_loss_factor = self.opacity_loss.forward(
            rendered_silhouettes,
            self.current_epoch,
        )

        ###########################################################################
        # Color loss
        ###########################################################################

        colors_at_rays = sample_images_at_mc_locs(
            images,
            ray_bundle.xys
        )
        color_err = self.loss_func(
            rendered_images,
            colors_at_rays,
        )

        ###########################################################################

        # Log the errors.
        self.log('color_loss', color_err, prog_bar=True, logger=True)
        self.log('sil_loss_world', sil_err_world, prog_bar=True, logger=True)
        self.log('sil_loss_can', sil_err_can, prog_bar=True, logger=True)
        self.log('opacity_loss', opacity_err, prog_bar=True, logger=True)

        # The optimization loss is a simple
        # sum of the color and silhouette errors.
        # TODO modified
        # loss = color_err + sil_err_world + sil_err_can + opacity_err
        loss = sil_err_can
        return loss


    def validation_step(self, batch, batch_idx):
        # result = self.visualize_batch(batch, warp_rays=True, cameras_canonical=False, canonical_renderer=False)

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

        ###############################################################
        # Ray sampling + warping
        ###############################################################

        ray_bundle = self.raysampler_val(
            cameras=batch_cameras,
        )
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        # Warp the rays to the canonical view.
        ray_points_can, ray_directions_can = warp_points.warp_points_batched(
            rays_points_world,
            manos['verts'],
            manos['Ts'],
        )

        ###########################################################################
        # Rendering
        ###########################################################################

        # get output of nerf model
        rays_densities, rays_features = self.neural_radiance_field.batched_forward(
            ray_points=ray_points_can, ray_directions=ray_directions_can
        )

        # render the images and silhouettes
        rendered_images_silhouettes = self.raymarcher(
            rays_densities=rays_densities,
            rays_features=rays_features,
        )
        assert rendered_images_silhouettes.isnan().any() == False
        
        # Split the rendering result to a silhouette and image render
        rendered_image, rendered_silhouette = (
            rendered_images_silhouettes[0].split([3, 1], dim=-1)
        )
        # convert rendered silhouette to rgb
        rendered_silhouette = torch.cat([rendered_silhouette, rendered_silhouette, rendered_silhouette], dim=-1)

        # concatenate the rendered image and the silhouette, transpose to (C, H, W)
        concat_rendered = torch.cat([rendered_image, rendered_silhouette], dim=1).permute(2, 0, 1)

        self.validation_images.append(concat_rendered)
        
        
    def validate_canonical_space(self, batch_cameras):
        ###############################################################
        # Ray sampling
        ###############################################################

        ray_bundle = self.raysampler_canonical(
            cameras=batch_cameras,
        )
        ray_points_can = ray_bundle_to_ray_points(ray_bundle)
        ray_directions_can_one_dir_per_ray = ray_bundle.directions

        # expand ray directions, from one direction per ray to one direction per each point
        spatial_size = ray_points_can.shape[:-1]
        ray_directions_can = ray_directions_can_one_dir_per_ray[..., None, :].expand(
            *spatial_size, ray_directions_can_one_dir_per_ray.shape[-1]
        )

        ###########################################################################
        # Rendering
        ###########################################################################

        # get output of nerf model
        rays_densities, rays_features = self.neural_radiance_field.batched_forward(
            ray_points=ray_points_can, ray_directions=ray_directions_can
        )

        # render the images and silhouettes
        rendered_images_silhouettes = self.raymarcher(
            rays_densities=rays_densities,
            rays_features=rays_features,
        )
        assert rendered_images_silhouettes.isnan().any() == False

        # Split the rendering result to a silhouette and image render
        rendered_image, rendered_silhouette = (
            rendered_images_silhouettes[0].split([3, 1], dim=-1)
        )
        # convert rendered silhouette to rgb
        rendered_silhouette = torch.cat([rendered_silhouette, rendered_silhouette, rendered_silhouette], dim=-1)

        # concatenate the rendered image and the silhouette, transpose to (C, H, W)
        concat_rendered = torch.cat([rendered_image, rendered_silhouette], dim=1).permute(2, 0, 1)

        self.can_validation_images.append(concat_rendered)
    

    def on_validation_epoch_end(self):
        grid = torchvision.utils.make_grid(self.validation_images, nrow=5)
        self.validation_images = []

        batch_cameras = create_canonical_cameras(10, random_cameras=False, device=self.device)
        for i in range(len(batch_cameras)):
            self.validate_canonical_space(batch_cameras[i])

        grid_can = torchvision.utils.make_grid(self.can_validation_images, nrow=5)
        self.can_validation_images = []

        tensorboard_logger = self.logger.experiment
        tensorboard_logger.add_image(f'validation_world', grid, self.current_epoch)
        tensorboard_logger.add_image(f'validation_can', grid_can, self.current_epoch)


    def get_nerf_output_canonical(self, points, directions):
        ray_densities, ray_colors = self.neural_radiance_field.forward_points(
            points,
            directions
        )
        return ray_densities, ray_colors

    def get_nerf_output_world(self, rays_points_world, verts, Ts):
        # Warp the rays to the canonical view.
        ray_points_can, ray_directions_can = warp_points.warp_points(
            rays_points_world,
            verts,
            Ts
        )
        ray_densities, ray_colors = self.neural_radiance_field.forward_points(
            ray_points_can,
            ray_directions_can
        )
        return ray_densities, ray_colors


    def get_nerf_output_legacy(self, points, directions):
        ray_densities, ray_colors = self.neural_radiance_field.forward_points(
            points,
            directions,
            None,
            None,
            warp_rays=False
        )
        return ray_densities, ray_colors
