def _smpl_shape_regularization(self, batch, pts, dirs, pred):
    raise NotImplementedError('TODO change to torch')

    device = pts.device
    smpl_reg = torch.tensor(0.0, requires_grad=True).float().to(device)
    can_mesh = self.val_dataset.scene.captures[batch['cap_id']].can_mesh

    # TODO change to torch
    dist_human, _, _ = igl.signed_distance(
        pts.reshape(-1, 3).detach().cpu().numpy(),
        can_mesh.verts_packed().cpu().numpy(),
        can_mesh.faces_packed().cpu().numpy(),
    )
    print('can_mesh', type(can_mesh))  # can_mesh.shape, can_mesh)
    print('pts.reshape(-1, 3)', pts.reshape(-1, 3).shape, pts.reshape(-1, 3))
    print('can_mesh.verts_packed()', can_mesh.verts_packed().shape, can_mesh.verts_packed())
    print('can_mesh.faces_packed()', can_mesh.faces_packed().shape, can_mesh.faces_packed())
    print('dist_human', dist_human.shape)
    print(dist_human)

    raise NotImplementedError('TODO change to torch')
    exit()

    inside_volume = dist_human < 0
    if inside_volume.sum() > 0:
        smpl_reg = smpl_reg + F.mse_loss(
            1 - torch.exp(-torch.relu(pred.reshape(-1, 4)[inside_volume][:, 3])),
            torch.ones_like(pred.reshape(-1, 4)[inside_volume][:, 3])
        ) * self.penalize_smpl_alpha

    # generate random samples inside a box in canonical space
    if self.penalize_dummy > 0:
        dummy_pts = (torch.rand(pts.shape, dtype=pts.dtype, device=device) - 0.5) * 3
        dummy_out = self.model.coarse_human_net(dummy_pts, dirs)

        # TODO change to torch
        dist_dummy, _, _ = igl.signed_distance(
            dummy_pts.reshape(-1, 3).detach().cpu().numpy(),
            can_mesh.verts_packed().cpu().numpy(),
            can_mesh.faces_packed().cpu().numpy(),
        )
        dummy_inside = dist_dummy < 0
        dummy_outside = dist_dummy > 0
        if dummy_inside.sum() > 0:
            smpl_reg = smpl_reg + F.mse_loss(
                1 - torch.exp(-torch.relu(dummy_out.reshape(-1, 4)[dummy_inside][:, 3])),
                torch.ones_like(dummy_out.reshape(-1, 4)[dummy_inside][:, 3])
            ) * self.penalize_dummy
        if dummy_outside.sum() > 0:
            smpl_reg = smpl_reg + F.l1_loss(
                (1 - torch.exp(-torch.relu(dummy_out.reshape(-1, 4)[dummy_outside][:, 3]))) * torch.pow(torch.abs(
                    torch.from_numpy(dist_dummy[dummy_outside]).to(device)) * self.opt.penalize_outside_factor,
                                                                                                        self.opt.dist_exponent),
                torch.zeros_like(dummy_out.reshape(-1, 4)[dummy_outside][:, 3])
            ) * self.penalize_dummy
    return smpl_reg