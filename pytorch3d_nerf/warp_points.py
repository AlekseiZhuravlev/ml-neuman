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
    return rays_points_world

    orig_rays_points_world = rays_points_world.clone()
    orig_shape = rays_points_world.shape

    if len(orig_shape) == 3:
        rays_points_world = rays_points_world.unsqueeze(0)

    rays_points_world = rays_points_world.reshape(rays_points_world.shape[0], -1, 3)

    print('rays_points_world', rays_points_world.shape)
    print('vertices', vertices.shape)
    print('Ts', Ts.shape)

    # get the transformation matrix of the closest vertex
    distances = torch.cdist(rays_points_world, vertices)
    print('distances', distances.shape)

    min_distances = torch.min(distances, dim=-1)[0]
    print('min_distances', torch.min(min_distances))

    res = torch.argmin(distances,dim=-1)
    print('res', res.shape)

    T_inv = torch.inverse(Ts)
    print('T_inv', T_inv.shape)

    # res torch.Size([1, 85504])
    # T_inv torch.Size([1, 778, 4, 4])
    # index T_inv with res
    Ts = T_inv[:, res, :, :].squeeze(0)

    # count how many elements have the same value in res
    res_unique, res_counts = torch.unique(res, return_counts=True)
    print('distances', distances)
    print('res_unique', res_unique, res_counts)
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
    print('ray_points_xyz_homo', ray_points_xyz_homo.shape)

    # Ts torch.Size([1, 342016, 4, 4])
    # ray_points_xyz torch.Size([1, 342016, 3])

    # for each batch, perform torch.bmm. Unsqueeze from [4] to [4, 1] vector
    ray_points_xyz_homo = ray_points_xyz_homo.unsqueeze(-1)
    can_pts_xyz = torch.stack([torch.bmm(Ts[i], ray_points_xyz_homo[i]) for i in range(Ts.shape[0])], dim=0)

    # Squeeze from [4, 1] to [4] vector, drop homogeneous dimension
    can_pts_xyz = can_pts_xyz.squeeze(-1)[..., :3]
    print('can_pts_xyz', can_pts_xyz.shape)

    # can_pts_xyz = (Ts @ ray_points_xyz_homo)  #[:, :3, 0].reshape(human_b, human_n, 3)
    # print(can_pts_xyz)
    # [B x N x 3]
    # get canonical directions
    can_dirs_xyz = can_pts_xyz[..., 1:] - can_pts_xyz[..., :-1]
    can_dirs_xyz = torch.cat([can_dirs_xyz, can_dirs_xyz[..., -1:]], dim=-1)
    can_dirs_xyz = can_dirs_xyz / torch.norm(can_dirs_xyz, dim=-1, keepdim=True)

    print('can_dirs_xyz', can_dirs_xyz.shape)


    can_pts_py3d = torch.stack(
        (-can_pts_xyz[..., 0], -can_pts_xyz[..., 1], can_pts_xyz[..., 2]),
        2
    )
    can_dirs_py3d = torch.stack(
        (-can_dirs_xyz[..., 0], -can_dirs_xyz[..., 1], can_dirs_xyz[..., 2]),
        2
    )

    # reshape can_pts and can_dirs to the input dimensions
    can_pts_py3d = can_pts_py3d.reshape(orig_shape)
    can_dirs_py3d = can_dirs_py3d.reshape(orig_shape)


    return can_pts_py3d, can_dirs_py3d