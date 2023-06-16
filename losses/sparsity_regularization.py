from utils.constant import HARD_SURFACE_OFFSET
import torch

def sparsity_regularization(
        can_mask,
        can_weights,
        penalize_sharp_edge,
        penalize_hard_surface,
):
    """
    Penalizes rendered mask and weights for sharp edges and hard surfaces
    """

    # make sure the weights and mask are in [0, 1]
    can_mask = torch.clip(can_mask, 0.0, 1.0)
    can_weights = torch.clip(can_weights, 0.0, 1.0)

    sparsity_reg = torch.tensor(0.0, requires_grad=True, device=can_mask.device)

    # sharp edge loss
    sparsity_reg = sparsity_reg + \
                   torch.mean(
                       -torch.log
                           (
                           torch.exp(-torch.abs(can_mask)) +
                           torch.exp(-torch.abs(1 - can_mask))
                       ) + HARD_SURFACE_OFFSET) * \
                   penalize_sharp_edge

    # hard surface loss
    sparsity_reg = sparsity_reg + \
                   torch.mean(
                       -torch.log
                           (
                           torch.exp(-torch.abs(can_weights)) +
                           torch.exp(-torch.abs(1 - can_weights))
                       ) + HARD_SURFACE_OFFSET) \
                   * penalize_hard_surface

    return sparsity_reg



def _sparsity_regularization_original(device):
    """
    Sample random straight rays in canonical space
    Get the output of the model
    Penalize
    """
    raise NotImplementedError

    sparsity_reg = torch.tensor(0.0, requires_grad=True, device=device)

    # pick a random camera
    num_can_rays = 128
    can_cap = random.choice(self.can_caps)

    # get coordinates of every pixel
    coords = torch.argwhere(torch.ones(can_cap.shape))

    # choose num_can_rays coordinates
    coords = coords[torch.random.randint(0, len(coords), num_can_rays)][:, ::-1]  # could get duplicated rays

    # shoot rays
    can_orig, can_dir = ray_utils.shot_rays(can_cap, coords)

    # sample points on the rays
    can_pts, can_dirs, can_z_vals = ray_utils.ray_to_samples(
        {
            'origin': torch.from_numpy(can_orig).float().to(device),
            'direction': torch.from_numpy(can_dir).float().to(device),
            'near': torch.zeros(num_can_rays, 1).float().to(device),
            'far': torch.ones(num_can_rays, 1).float().to(device) * CANONICAL_CAMERA_DIST * 1.667,
        },
        samples_per_ray=self.opt.samples_per_ray,
        device=device,
        perturb=self.opt.perturb
    )

    # run the model
    can_out = self.model.coarse_human_net(can_pts, can_dirs)

    # compensate the interval difference between observation space and canonical space
    can_out[..., -1] *= self.interval_comp

    # render the output
    _, _, can_mask, can_weights, _ = render_utils.raw2outputs(can_out, can_z_vals.clone(), can_dirs[:, 0, :].clone(),
                                                              white_bkg=True)

    # make sure the weights and mask are in [0, 1]
    can_weights = torch.clip(can_weights, 0.0, 1.0)
    can_mask = torch.clip(can_mask, 0.0, 1.0)

    # sharp edge loss
    if self.penalize_sharp_edge > 0:
        sparsity_reg = sparsity_reg + \
                       torch.mean(
                           -torch.log
                               (
                               torch.exp(-torch.abs(can_mask)) +
                               torch.exp(-torch.abs(1 - can_mask))
                           ) + HARD_SURFACE_OFFSET) * \
                       self.penalize_sharp_edge

    # hard surface loss
    if self.penalize_hard_surface > 0:
        sparsity_reg = sparsity_reg + \
                       torch.mean(
                           -torch.log
                               (
                               torch.exp(-torch.abs(can_weights)) +
                               torch.exp(-torch.abs(1 - can_weights))
                           ) + HARD_SURFACE_OFFSET) \
                       * self.penalize_hard_surface

    return sparsity_reg
