import torch
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
import sampling_utils


class RendererWarp:
    def __init__(
            self,
            use_offset_net
    ):
        self.raymarcher = EmissionAbsorptionRaymarcher()
        self.use_offset_net = use_offset_net

    def forward(
            self,
            raysampler,
            batch_cameras,
            verts,
            Ts,
            masks_sampling,
            nerf_func,
            warp_func,
            offset_net_func,
            curr_pose_id,
            logger # fixme: debug only
            ):

        ###############################################################
        # Get depths and masks for ray sampling
        ###############################################################

        depths = batch_cameras.get_world_to_view_transform().transform_points(
            verts
        )[:, :, 2:]

        ###############################################################
        # Ray sampling in world space + warping
        ###############################################################

        ray_bundle = raysampler(
            cameras=batch_cameras,
            mask=masks_sampling,
            min_depth=depths.min() * 0.95,
            max_depth=depths.max() * 1.05,
        )
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        if warp_func:
            # Warp the rays to the canonical view.
            ray_points_can, ray_directions_can = warp_func(
                rays_points_world,
                verts,
                Ts,
            )
            if self.use_offset_net:
                # predict offset
                curr_pose_id_vec = torch.ones_like(ray_points_can[..., 0:1]) * curr_pose_id
                offset = offset_net_func(torch.cat([ray_points_can, curr_pose_id_vec], dim=-1))
                logger.log('offset_mean', offset.mean())

                ray_points_can += offset
        else:
            # no warping
            ray_points_can = rays_points_world
            ray_directions_can_one_dir_per_ray = ray_bundle.directions

            # expand ray directions, from one direction per ray to one direction per each point
            spatial_size = ray_points_can.shape[:-1]
            ray_directions_can = ray_directions_can_one_dir_per_ray[..., None, :].expand(
                *spatial_size, ray_directions_can_one_dir_per_ray.shape[-1]
            )

        assert ray_points_can.isnan().any() == False
        assert ray_directions_can.isnan().any() == False

        ###########################################################################
        # Rendering
        ###########################################################################

        # get output of nerf model
        rays_densities, rays_features = nerf_func(
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

        return rendered_images, rendered_silhouettes, ray_bundle