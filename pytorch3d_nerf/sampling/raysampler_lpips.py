import warnings
from typing import Optional, Tuple, Union

import torch
from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.ops import padded_to_packed
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.utils import HeterogeneousRayBundle, RayBundle
from pytorch3d.renderer.implicit.raysampling import _xy_to_ray_bundle, _pack_ray_bundle
from torch.nn import functional as F

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

class LPIPSRaysampler(NDCMultinomialRaysampler):
    def forward_lpips(
            self,
            cameras: CamerasBase,
            mask,
            min_depth,
            max_depth,
    ) -> Union[RayBundle, HeterogeneousRayBundle]:

        assert mask.sum() == self._n_rays_per_image, f'mask.sum() = {mask.sum()}, _n_rays_per_image = {self._n_rays_per_image}'

        camera_ids: torch.LongTensor = torch.arange(len(cameras), dtype=torch.long)
        batch_size = cameras.R.shape[0]
        device = cameras.device

        # expand the (H, W, 2) grid batch_size-times to (B, H, W, 2)
        xy_grid = self._xy_grid.to(device).expand(batch_size, -1, -1, -1)


        xy_grid_flat = xy_grid.reshape(batch_size, -1, 2)
        mask_flat = mask.reshape(batch_size, -1)

        xy_grid_indexed = xy_grid_flat[mask_flat.bool()]

        mask_sum = mask_flat[0].sum()
        mask_side = torch.sqrt(mask_sum).int()
        assert mask_side * mask_side == mask_sum, f'mask_side = {mask_side}, mask_sum = {mask_sum}'

        xy_grid_indexed_reshaped = xy_grid_indexed.reshape(batch_size, mask_side, mask_side, 2)

        ray_bundle = _xy_to_ray_bundle(
            cameras,
            xy_grid_indexed_reshaped,
            min_depth,
            max_depth,
            self._n_pts_per_ray,
            self._unit_directions,
            self._stratified_sampling,
        )

        return ray_bundle