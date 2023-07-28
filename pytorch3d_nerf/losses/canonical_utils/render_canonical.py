import torch
from render_utils.render_mesh import render_mesh
from render_utils.render_point_cloud import render_point_cloud

class RendererCanonical:

    def render_zero_pose(self, cameras, verts_py3d, faces):
        n_cameras = len(cameras)

        # copy verts_py3d into 1st dimension of batch, number = 10
        verts_py3d = verts_py3d.repeat(n_cameras, 1, 1)
        faces = faces.repeat(n_cameras, 1, 1)

        img, depth = render_mesh(verts_py3d, faces, cameras, no_grad=False)
        return img, depth


    def render_nerf_point_cloud(self, cameras, rays_points_can, rays_densities, rays_features):
        n_cameras = len(cameras)

        points_reshaped = rays_points_can.reshape(1, -1, 3).repeat(n_cameras, 1, 1)
        rays_colors_reshaped = rays_features.reshape(1, -1, 3)
        rays_densities_reshaped = rays_densities.reshape(1, -1, 1)

        rays_colors_rgba = torch.cat((rays_colors_reshaped, rays_densities_reshaped), axis=-1).repeat(n_cameras, 1, 1)

        img, depth = render_point_cloud(points_reshaped, rays_colors_rgba, cameras, no_grad=False, background_color=(0, 0, 0))
        return img, depth


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
