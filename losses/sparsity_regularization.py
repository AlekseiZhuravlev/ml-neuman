
def _sparsity_regularization(self, device):
    raise NotImplementedError('TODO change to torch')

    sparsity_reg = torch.tensor(0.0, requires_grad=True, device=device)
    # pick a random camera
    num_can_rays = 128
    can_cap = random.choice(self.can_caps)
    coords = torch.argwhere(torch.ones(can_cap.shape))
    coords = coords[torch.random.randint(0, len(coords), num_can_rays)][:, ::-1]  # could get duplicated rays
    can_orig, can_dir = ray_utils.shot_rays(can_cap, coords)
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
    can_out = self.model.coarse_human_net(can_pts, can_dirs)
    # compensate the interval difference between observation space and canonical space
    can_out[..., -1] *= self.interval_comp
    _, _, can_mask, can_weights, _ = render_utils.raw2outputs(can_out, can_z_vals.clone(), can_dirs[:, 0, :].clone(),
                                                              white_bkg=True)
    can_weights = torch.clip(can_weights, 0.0, 1.0)
    can_mask = torch.clip(can_mask, 0.0, 1.0)

    # sharp edge loss
    if self.penalize_sharp_edge > 0:
        sparsity_reg = sparsity_reg + torch.mean(-torch.log(
            torch.exp(-torch.abs(can_mask)) + torch.exp(-torch.abs(1 - can_mask))
        ) + HARD_SURFACE_OFFSET) * self.penalize_sharp_edge

    # hard surface loss
    if self.penalize_hard_surface > 0:
        sparsity_reg = sparsity_reg + torch.mean(-torch.log(
            torch.exp(-torch.abs(can_weights)) + torch.exp(-torch.abs(1 - can_weights))
        ) + HARD_SURFACE_OFFSET) * self.penalize_hard_surface

    return sparsity_reg