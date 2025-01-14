import math
from typing import Optional, Tuple

import pytorch3d

import torch
from pytorch3d.ops import packed_to_padded
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures import Pointclouds

from pytorch3d.implicitron.tools.rasterize_mc import rasterize_mc_samples


def rasterize_sparse_ray_bundle(
    ray_bundle,
    features: torch.Tensor,
    image_size_hw: Tuple[int, int],
    depth: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rasterizes sparse features corresponding to the coordinates defined by
    the rays in the bundle.

    Args:
        ray_bundle: ray bundle object with B x ... x 2 pixel coordinates,
            it can be packed.
        features: B x ... x C tensor containing per-point rendered features.
        image_size_hw: Tuple[image_height, image_width] containing
            the size of rasterized image.
        depth: B x ... x 1 tensor containing per-point rendered depth.
        masks: B x ... x 1 tensor containing the alpha mask of the
            rendered features.

    Returns:
        - image_render: B x C x H x W tensor of rasterized features
        - depths_render: B x 1 x H x W tensor of rasterized depth maps
        - masks_render: B x 1 x H x W tensor of opacities after splatting
    """
    # Flatten the features and xy locations.
    features_depth_ras = torch.cat(
        (features.flatten(1, -2), depth.flatten(1, -2)), dim=-1
    )
    xys = ray_bundle.xys
    masks_ras = None
    # if ray_bundle.is_packed():
    #     camera_counts = ray_bundle.camera_counts
    #     assert camera_counts is not None
    #     xys, first_idxs, _ = ray_bundle.get_padded_xys()
    #     masks_ras = (
    #         torch.arange(xys.shape[1], device=xys.device)[:, None]
    #         < camera_counts[:, None, None]
    #     )
    #
    #     max_size = torch.max(camera_counts).item()
    #     features_depth_ras = packed_to_padded(
    #         features_depth_ras[:, 0], first_idxs, max_size
    #     )
    #     if masks is not None:
    #         padded_mask = packed_to_padded(masks.flatten(1, -1), first_idxs, max_size)
    #         masks_ras = padded_mask * masks_ras

    xys_ras = xys.flatten(1, -2)

    if masks_ras is None:
        # assert not ray_bundle.is_packed()
        masks_ras = masks.flatten(1, -2) if masks is not None else None

    if min(*image_size_hw) <= 0:
        raise ValueError(
            "Need to specify a positive output_size_hw for bundle rasterisation."
        )

    # Estimate the rasterization point radius so that we approximately fill
    # the whole image given the number of rasterized points.
    pt_radius = 2.0 / math.sqrt(xys.shape[1])

    # Rasterize the samples.
    features_depth_render, masks_render = rasterize_mc_samples(
        xys_ras,
        features_depth_ras,
        image_size_hw,
        radius=pt_radius,
        masks=masks_ras,
    )
    images_render = features_depth_render[:, :-1]
    depths_render = features_depth_render[:, -1:]
    return images_render, depths_render, masks_render