import torch

def warp_points(
        rays_points_world,
        vertices,
        Ts,
):
    """
        rays_points_world torch.Size([2672, 32, 3]) OR torch.Size([1, 2048, 128, 3])
        vertices torch.Size([1, 778, 3])
        Ts torch.Size([1, 778, 4, 4])
    """
    orig_shape = rays_points_world.shape

    if len(orig_shape) == 3:
        rays_points_world = rays_points_world.unsqueeze(0)

    print('rays_points_world', rays_points_world.shape)
    print('vertices', vertices.shape)
    print('Ts', Ts.shape)

    # get the transformation matrix of the closest vertex
    res = torch.cdist(rays_points_world, vertices)
    print('res', res.shape)

    res = torch.argmin(res,dim=-1)
    print('res', res.shape)

    T_inv = torch.inverse(Ts)
    print('T_inv', T_inv.shape)

    # invert the transformation matrix
    Ts = T_inv[res]
    # access res torch.Size([2672, 32])
    # T_inv torch.Size([1, 778, 4, 4])
    # without indexing the tensor
    # Ts = torch.gather

    print('Ts', Ts.shape)

    # convert the points from pytorch3d space to xyz space
    ray_points_xyz = torch.stack(
        (-rays_points_world[..., 0], -rays_points_world[..., 1], rays_points_world[..., 2]),
        -1
    )
    print('ray_points_xyz', ray_points_xyz.shape)

    # add homogeneous coordinate
    ray_points_xyz_homo = torch.cat([
        ray_points_xyz,
        torch.ones_like(ray_points_xyz[..., 0:1])
    ], axis=-1)

    can_pts_xyz = (Ts @ ray_points_xyz_homo)[:, :3, 0].reshape(human_b, human_n, 3)

    can_pts_py3d = torch.stack(
        (-can_pts_xyz[..., 0], -can_pts_xyz[..., 1], can_pts_xyz[..., 2]),
        2
    )

    exit()

    # get canonical directions
    can_dirs = can_pts[:, 1:] - can_pts[:, :-1]
    can_dirs = torch.cat([can_dirs, can_dirs[:, -1:]], dim=1)
    can_dirs = can_dirs / torch.norm(can_dirs, dim=2, keepdim=True)

    return

    return rays_points_world