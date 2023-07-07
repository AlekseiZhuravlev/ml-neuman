import warnings
from typing import Optional, Tuple, Union

import torch
from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.ops import padded_to_packed
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.utils import HeterogeneousRayBundle, RayBundle
from torch.nn import functional as F


class MonteCarloRaysamplerMasked(pytorch3d.renderer.NDCMultinomialRaysampler):
    """
    Same as MonteCarloRaysampler but with a mask to sample only a subset of the image
    """
    def __init__(
            *args,
            n_rays_per_image: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # self.n

    def forward(
        self,
        cameras: CamerasBase,
        mask = None,
        *,
        stratified_sampling: Optional[bool] = None,
        **kwargs,
    ) -> Union[RayBundle, HeterogeneousRayBundle]:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            stratified_sampling: if set, overrides stratified_sampling provided

        """
        if (
            sum(x is not None for x in [self._n_rays_total, self._n_rays_per_image])
            != 1
        ):
            raise ValueError(
                "Exactly one of `self.n_rays_total` and `self.n_rays_per_image` "
                "must be given."
            )

        if self._n_rays_total:
            (
                cameras,
                _,
                camera_ids,
                camera_counts,
                n_rays_per_image,
            ) = _sample_cameras_and_masks(self._n_rays_total, cameras, None)
        else:
            # pyre-ignore[9]
            camera_ids: torch.LongTensor = torch.arange(len(cameras), dtype=torch.long)
            n_rays_per_image = self._n_rays_per_image

        batch_size = cameras.R.shape[0]

        device = cameras.device

        # get the initial grid of image xy coords
        # of shape (batch_size, n_rays_per_image, 2)
        if mask:
            rays_xy = None
        else:
            rays_xy = torch.cat(
                [
                    torch.rand(
                        size=(batch_size, n_rays_per_image, 1),
                        dtype=torch.float32,
                        device=device,
                    )
                    * (high - low)
                    + low
                    for low, high in (
                        (self._min_x, self._max_x),
                        (self._min_y, self._max_y),
                    )
                ],
                dim=2,
            )

        stratified_sampling = (
            stratified_sampling
            if stratified_sampling is not None
            else self._stratified_sampling
        )

        ray_bundle = _xy_to_ray_bundle(
            cameras,
            rays_xy,
            self._min_depth,
            self._max_depth,
            self._n_pts_per_ray,
            self._unit_directions,
            stratified_sampling,
        )

        return (
            # pyre-ignore[61]
            _pack_ray_bundle(ray_bundle, camera_ids, camera_counts)
            if self._n_rays_total
            else ray_bundle
        )