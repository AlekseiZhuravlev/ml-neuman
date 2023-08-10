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
import sampling_utils
import warp_points
from helpers import sample_images_at_mc_locs
from losses import huber
from losses.canonical_utils.cameras_canonical import create_canonical_cameras
from losses.opacity_loss import OpacityLoss
from losses.sil_loss_can import SilhouetteLossCanonical
from losses.sil_loss_world import SilhouetteLossWorld
from mano_custom import mano_pytorch3d
from renderers.renderer_warp import RendererWarp
import warp_points


class HandModel(L.LightningModule):
    def __init__(self,
                 nerf_model,
                 offset_net,
                 loss_func_color,
                 sil_loss_world,
                 sil_loss_can,
                 renderer_warp,
                 enable_offset_net
                 ):
        super().__init__()

        self.hand_model = mano_pytorch3d.create_mano_custom(return_right_hand=False)
        self.register_buffer(
            'faces',
            torch.from_numpy(self.hand_model.faces.astype(np.int32))[None, :, :]
        )

        # TODO this is not 100% accurate, but it's close enough
        self.register_buffer(
            'verts_zero_pose',
            self.hand_model.get_flat_hand_vertices_pytorch3d(self.device)
        )

        self.min_depth = 0.1
        self.max_depth = 2

        self.render_size_x = 512
        self.render_size_y = 334

        self.raysampler_train = NDCMultinomialRaysampler(
            image_height=self.render_size_x,
            image_width=self.render_size_y,
            n_rays_per_image=4096,
            n_pts_per_ray=64,
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
        self.renderer_warp = renderer_warp

        # Instantiate the radiance field model.
        self.neural_radiance_field = nerf_model
        self.offset_net = offset_net

        if not enable_offset_net:
            # set offset net parameters as non-trainable
            for param in self.offset_net.parameters():
                param.requires_grad = False

        self.validation_images = []
        self.can_validation_images = []

        self.loss_func_color = loss_func_color
        self.sil_loss_world = sil_loss_world
        self.sil_loss_can = sil_loss_can


    def configure_optimizers(self):
        print('self.neural_radiance_field', self.neural_radiance_field)
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
        camera_params, images, silhouettes, silhouettes_can, manos = batch

        ###########################################################################
        # Raysampling in world space
        ###########################################################################

        batch_cameras = PerspectiveCameras(
            R=camera_params['R_pytorch3d'],
            T=camera_params['t_pytorch3d'],
            focal_length=camera_params['focal'],
            principal_point=camera_params['princpt'],
            in_ndc=False,
            image_size=camera_params['image_size'],
            device=self.device
        )
        rendered_images, rendered_silhouettes, ray_bundle = self.renderer_warp.forward(
            raysampler=self.raysampler_train,
            batch_cameras=batch_cameras,
            verts=manos['verts'],
            Ts=manos['Ts'],

            masks_sampling=sampling_utils.sampling_mask_0_25(silhouettes),
            nerf_func=self.neural_radiance_field.forward,
            warp_func=warp_points.warp_points,

            offset_net_func=self.offset_net.forward,
            curr_pose_id=manos['pose_id'],
            logger=self.logger.experiment,
        )  # rendered_images_silhouettes.shape torch.Size([1, 8192, 1, 4])

        ###########################################################################
        # Raysampling in canonical space
        ###########################################################################

        batch_cameras_can = FoVPerspectiveCameras(
            R=camera_params['R_can'],
            T=camera_params['t_can'],
            znear=0.01,
            zfar=10,
            device=self.device,
        )

        rendered_images_can, rendered_silhouettes_can, ray_bundle_can = self.renderer_warp.forward(
            raysampler=self.raysampler_train,
            batch_cameras=batch_cameras_can,
            verts=manos['verts_zero'],
            Ts=None,

            masks_sampling=sampling_utils.sampling_mask_25_0_25(silhouettes_can),
            nerf_func=self.neural_radiance_field.forward,
            warp_func=None,

            offset_net_func=None,
            curr_pose_id=None,
            logger=None,
        )

        ###########################################################################
        # Silhouette loss, canonical space
        ###########################################################################

        sil_err_can, sil_err_unconstrained_can, sil_loss_factor_can = self.sil_loss_can.forward(
            rendered_silhouettes_can,
            silhouettes_can,
            ray_bundle_can,
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

        ###########################################################################
        # Color loss
        ###########################################################################

        colors_at_rays = pytorch3d.renderer.utils.ndc_grid_sample(
            images.permute(0, 3, 1, 2),
            ray_bundle.xys,
        ).permute(0, 2, 3, 1)

        color_err = self.loss_func_color(
            rendered_images,
            colors_at_rays,
        )

        ###########################################################################

        # Log the errors.
        self.log('color_loss', color_err, prog_bar=True, logger=True)
        self.log('sil_loss_world', sil_err_world, prog_bar=True, logger=True)
        self.log('sil_loss_can', sil_err_can, prog_bar=True, logger=True)

        loss = color_err + sil_err_world + sil_err_can
        return loss


    def validation_step(self, batch, batch_idx):
        # result = self.visualize_batch(batch, warp_rays=True, cameras_canonical=False, canonical_renderer=False)

        camera_params, images, silhouettes, silhouettes_zero, manos = batch

        batch_cameras = PerspectiveCameras(
            R=camera_params['R_pytorch3d'],
            T=camera_params['t_pytorch3d'],
            focal_length=camera_params['focal'],
            principal_point=camera_params['princpt'],
            in_ndc=False,
            image_size=camera_params['image_size'],
            device=self.device
        )

        rendered_image, rendered_silhouette, ray_bundle = self.renderer_warp.forward(
            raysampler=self.raysampler_canonical,
            batch_cameras=batch_cameras,
            verts=manos['verts'],
            Ts=manos['Ts'],

            masks_sampling=None,
            nerf_func=self.neural_radiance_field.batched_forward,
            warp_func=warp_points.warp_points_batched,

            offset_net_func=self.offset_net.batched_forward,
            curr_pose_id=manos['pose_id'],
            logger=self.logger.experiment,
        )  # rendered_images_silhouettes.shape torch.Size([1, 8192, 1, 4])

        # convert rendered silhouette to rgb
        rendered_silhouette = torch.cat([rendered_silhouette, rendered_silhouette, rendered_silhouette], dim=-1)

        # concatenate the rendered image and the silhouette, transpose to (B, C, H, W)
        concat_rendered = torch.cat([rendered_image, rendered_silhouette], dim=1).permute(0, 3, 1, 2)

        self.validation_images.append(concat_rendered[0])


    def validate_canonical_space(self, batch_cameras):
        rendered_image, rendered_silhouette, ray_bundle = self.renderer_warp.forward(
            raysampler=self.raysampler_canonical,
            batch_cameras=batch_cameras,
            verts=self.verts_zero_pose,
            Ts=None,

            masks_sampling=None,
            nerf_func=self.neural_radiance_field.batched_forward,
            warp_func=None,

            offset_net_func=None,
            curr_pose_id=None,
            logger=self.logger.experiment,
        )  # rendered_images_silhouettes.shape torch.Size([1, 8192, 1, 4])

        # convert rendered silhouette to rgb
        rendered_silhouette = torch.cat([rendered_silhouette, rendered_silhouette, rendered_silhouette], dim=-1)

        # concatenate the rendered image and the silhouette, transpose to (B, C, H, W)
        concat_rendered = torch.cat([rendered_image, rendered_silhouette], dim=1).permute(0, 3, 1, 2)

        self.can_validation_images.append(concat_rendered[0])


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


    def on_fit_start(self):
        """
        add train and val images to tensorboard
        """

        train_images = torch.tensor(self.trainer.datamodule.train_dataset.images).permute(0, 3, 1, 2)
        train_sil = torch.tensor(self.trainer.datamodule.train_dataset.silhouettes).unsqueeze(-1).permute(0, 3, 1, 2)

        val_images = torch.tensor(self.trainer.datamodule.test_dataset.images).permute(0, 3, 1, 2)
        val_sil = torch.tensor(self.trainer.datamodule.test_dataset.silhouettes).unsqueeze(-1).permute(0, 3, 1, 2)

        train_images_grid = torchvision.utils.make_grid(train_images, nrow=5)
        train_sil_grid = torchvision.utils.make_grid(train_sil, nrow=5)
        val_images_grid = torchvision.utils.make_grid(val_images, nrow=5)
        val_sil_grid = torchvision.utils.make_grid(val_sil, nrow=5)

        tensorboard_logger = self.logger.experiment
        tensorboard_logger.add_image(f'train_images', train_images_grid, 0)
        tensorboard_logger.add_image(f'train_sil', train_sil_grid, 0)
        tensorboard_logger.add_image(f'val_images', val_images_grid, 0)
        tensorboard_logger.add_image(f'val_sil', val_sil_grid, 0)


    def get_nerf_output_canonical(self, points, directions):
        ray_densities, ray_colors = self.neural_radiance_field.forward_points(
            points,
            directions
        )
        return ray_densities, ray_colors

    def get_nerf_output_world(self, rays_points_world, verts, Ts):
        raise NotImplementedError
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
        return self.get_nerf_output_canonical(points, directions)
