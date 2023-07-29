import torch
import torch.nn as nn
from pytorch3d.renderer import (
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
    PointsRasterizationSettings,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
    SoftSilhouetteShader,
    BlendParams,
)
from pytorch3d.structures import Meshes, Pointclouds
# from render_utils.render_mesh import render_mesh
from render_utils.render_point_cloud import render_point_cloud
import lightning as L


class RendererCanonical(L.LightningModule):

    def __init__(self, cameras):
        super().__init__()
        self.n_cameras = len(cameras)

        # mesh renderer
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings_mesh = RasterizationSettings(
            image_size=(512, 334),
            blur_radius=0.0,
            faces_per_pixel=10,
        )
        self.renderer_mesh = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings_mesh
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )
        # point cloud renderer
        raster_settings = PointsRasterizationSettings(
            image_size=(512, 334),
            radius=0.01,
            points_per_pixel=10
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

        self.renderer_point_cloud = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=(0,0,0))
        )


    def render_zero_pose_sil(self, verts_py3d, faces):
        # copy verts_py3d into 1st dimension of batch, number = 10
        verts_py3d_repeated = verts_py3d.repeat(self.n_cameras, 1, 1)
        faces_repeated = faces.repeat(self.n_cameras, 1, 1)

        device = verts_py3d_repeated.device
        batch_size, vertex_num = verts_py3d_repeated.shape[:2]

        textures = TexturesVertex(verts_features=torch.ones((batch_size, vertex_num, 3)).float().to(device))
        meshes = Meshes(verts_py3d_repeated, faces_repeated, textures)

        silhouettes_rgba = self.renderer_mesh(meshes)
        silhouettes = (silhouettes_rgba[:, :, :, 3] > 0.1).unsqueeze(-1).float()
        return silhouettes

    def render_nerf_point_cloud(self, rays_points_can, rays_densities, rays_features):

        points_reshaped = rays_points_can.reshape(1, -1, 3).repeat(self.n_cameras, 1, 1)
        rays_colors_reshaped = rays_features.reshape(1, -1, 3)
        rays_densities_reshaped = rays_densities.reshape(1, -1, 1)

        rays_colors_rgba = torch.cat((rays_colors_reshaped, rays_densities_reshaped), axis=-1).repeat(self.n_cameras, 1, 1)

        point_clouds = Pointclouds(points=points_reshaped, features=rays_colors_rgba)

        images_depths = self.renderer_point_cloud(point_clouds)

        depthmaps = images_depths[:, :, :, 3].unsqueeze(-1)
        images = images_depths[:, :, :, :3]
        return images, depthmaps


    def render_nerf_point_cloud_threshold(self, cameras, rays_points_can, rays_densities, rays_features):
        points_reshaped = rays_points_can.reshape(-1, 3)
        rays_colors_reshaped = rays_features.reshape(-1, 3)
        rays_densities_reshaped = rays_densities.reshape(-1, 1)

        threshold = 0.01

        points_reshaped = points_reshaped[rays_densities.reshape(-1) > threshold]
        rays_colors_reshaped = rays_colors_reshaped[rays_densities.reshape(-1) > threshold]
        rays_densities_reshaped = rays_densities_reshaped[rays_densities.reshape(-1) > threshold]

        rays_colors_rgba = torch.cat((rays_colors_reshaped, rays_densities_reshaped), axis=-1)

        points_reshaped = points_reshaped.reshape(1, -1, 3)
        rays_colors_rgba = rays_colors_rgba.reshape(1, -1, 4)

        points_reshaped = points_reshaped.repeat(n_cameras, 1, 1)
        rays_colors_rgba = rays_colors_rgba.repeat(n_cameras, 1, 1)

        img, depth = render_point_cloud(points_reshaped, rays_colors_rgba, cameras)
        return img, depth
