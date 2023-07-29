import torch
import torch.nn as nn
from helpers import sample_images_at_mc_locs
from losses.canonical_utils.cameras_canonical import create_canonical_cameras
from losses.canonical_utils.render_canonical import RendererCanonical
import torchvision
import lightning as L


class SilhouetteLossCanonical(L.LightningModule):
    def __init__(self, n_cameras, verts_zero_pose, faces, loss_func, sil_loss_start_factor, sil_loss_epochs, device):
        super().__init__()

        self.n_cameras = n_cameras
        self.verts_zero = verts_zero_pose

        self.loss_func = loss_func
        self.sil_loss_start_factor = sil_loss_start_factor
        self.sil_loss_epochs = sil_loss_epochs

        # self.register_buffer(
        #     'cameras',
        #     create_canonical_cameras(self.n_cameras, random_cameras=False, device=verts_zero_pose.device)
        # )
        self.cameras = create_canonical_cameras(self.n_cameras, random_cameras=False, device=device)

        self.renderer = RendererCanonical(self.cameras)

        print(self.device)
        # print('renderer.device', self.renderer.device)
        # print('self.cameras.device', self.cameras.device)
        # print('verts_zero_pose.device', verts_zero_pose.device)
        # print('faces.device', faces.device)
        silhouettes_zero_pose = self.renderer.render_zero_pose_sil(
            verts_zero_pose,
            faces
        )
        self.register_buffer(
            'silhouettes_zero_pose',
            silhouettes_zero_pose
        )

    def forward(self,
                rays_points_can,
                rays_densities,
                rays_features,
                current_epoch,
                ):

        # print('self.cameras.device', self.cameras.device)

        img_nerf_point_cloud, depth_nerf_point_cloud = self.renderer.render_nerf_point_cloud(
            rays_points_can,
            rays_densities,
            rays_features
        )
        # print('depth_nerf_point_cloud.shape', depth_nerf_point_cloud.shape)
        sil_err_unconstrained = self.loss_func(depth_nerf_point_cloud, self.silhouettes_zero_pose)
        # print('sil_err_unconstrained', sil_err_unconstrained)
        # exit()

        # decrease silhouette loss and update the factor
        if self.sil_loss_epochs > 0:
            sil_loss_factor = self.sil_loss_start_factor * max(0, 1 - (current_epoch / self.sil_loss_epochs))
        else:
            sil_loss_factor = 0
        sil_err = sil_err_unconstrained * sil_loss_factor

        return sil_err, sil_err_unconstrained, sil_loss_factor



    def save_can_images(self, img_zero_pose, depth_zero_pose, img_nerf_point_cloud, depth_nerf_point_cloud):

        # concatenate image files, make depth 3 channels, switch n of channels
        img = torch.cat((img_zero_pose, img_nerf_point_cloud), dim=0).permute(0, 3, 1, 2)
        depth = torch.cat((depth_zero_pose, depth_nerf_point_cloud), dim=0).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)

        # make grid of images and save them
        grid_img = torchvision.utils.make_grid(img, nrow=10)
        torchvision.utils.save_image(grid_img,
                                     "/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/losses/canonical_utils/images/grid_images.png")

        # make grid of depth maps and save them
        grid_depth = torchvision.utils.make_grid(depth, nrow=10)
        torchvision.utils.save_image(grid_depth,
                                     "/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/losses/canonical_utils/images/grid_depths.png")

