import torch

class WarpCanonical(torch.nn.Module):
    @staticmethod
    def warp_points(
            rays_points_world,
            rays_directions_world,
            vertices_posed,
            Ts,
    ):
        """
        Warps the rays_points_world to the canonical space using the T of the closest vertex
        DO NOT MODIFY, USE DEBUG VERSION INSTEAD
        """
        orig_rays_points_world = rays_points_world.clone()

        if len(orig_rays_points_world.shape) == 3:
            rays_points_world = rays_points_world.unsqueeze(0)

        orig_shape = rays_points_world.shape

        rays_points_world = rays_points_world.reshape(1, -1, 3)

        # get the transformation matrix of the closest vertex
        distances = torch.cdist(rays_points_world, vertices_posed)
        res = torch.argmin(distances,dim=-1)

        T_inv = torch.inverse(Ts)

        # res torch.Size([1, 85504])
        res = res.squeeze(0)

        # T_inv torch.Size([1, 778, 4, 4])
        # index T_inv with res
        T_inv_vert = T_inv[:, res, :, :]

        # convert the points from pytorch3d space to xyz space
        ray_points_xyz = torch.stack(
            (-rays_points_world[..., 0], -rays_points_world[..., 1], rays_points_world[..., 2]),
            -1
        )
        # add homogeneous coordinate
        ray_points_xyz_homo = torch.cat([
            ray_points_xyz,
            torch.ones_like(ray_points_xyz[..., 0:1])
        ], axis=-1)

        # Ts torch.Size([1, 342016, 4, 4])
        # ray_points_xyz torch.Size([1, 342016, 3])

        # for each batch, perform torch.bmm. Unsqueeze from [4] to [4, 1] vector
        ray_points_xyz_homo = ray_points_xyz_homo.unsqueeze(-1)
        can_pts_xyz = torch.stack([torch.bmm(T_inv_vert[i], ray_points_xyz_homo[i]) for i in range(T_inv_vert.shape[0])], dim=0)

        # Squeeze from [4, 1] to [4] vector, drop homogeneous dimension
        can_pts_xyz = can_pts_xyz.squeeze(-1)[..., :3]

        # reshape can_pts to the input dimensions [B x N x 3] -> [B x N_rays x N_samples x 3]
        can_pts_xyz = can_pts_xyz.reshape(orig_shape)

        # canonical direction = direction from previous ray sample to current
        can_dirs_xyz = can_pts_xyz[..., 1:, :] - can_pts_xyz[..., :-1, :]
        can_dirs_xyz = torch.cat([can_dirs_xyz, can_dirs_xyz[..., -1:, :]], dim=-2)
        can_dirs_xyz = torch.nn.functional.normalize(
            can_dirs_xyz, dim=-1
        )

        can_pts_py3d = torch.stack(
            (-can_pts_xyz[..., 0], -can_pts_xyz[..., 1], can_pts_xyz[..., 2]),
            -1
        )
        can_dirs_py3d = torch.stack(
            (-can_dirs_xyz[..., 0], -can_dirs_xyz[..., 1], can_dirs_xyz[..., 2]),
            -1
        )

        # reshape can_pts and can_dirs to the input dimensions
        if len(orig_rays_points_world.shape) == 3:
            can_pts_py3d = can_pts_py3d.squeeze(0)
            can_dirs_py3d = can_dirs_py3d.squeeze(0)

        return can_pts_py3d, can_dirs_py3d

    @staticmethod
    def warp_points_batched(
            rays_points_world,
            rays_directions_world,
            vertices_posed,
            Ts,
            n_batches: int = 16,
        ):
            """
            Args:
                ray_points: torch.Size([1, 8192, 1, 32, 3])
                ray_directions: torch.Size([1, 8192, 1, 32, 3])
                n_batches: int = 16
            """
            batches_ray_points = torch.chunk(rays_points_world, chunks=n_batches, dim=1)
            batches_ray_dirs = torch.chunk(rays_directions_world, chunks=n_batches, dim=1)

            # For each batch, execute the standard forward pass and concatenate
            can_pts_py3d = torch.tensor([], device=rays_points_world.device)
            can_dirs_py3d = torch.tensor([], device=rays_points_world.device)

            for batch_idx in range(len(batches_ray_points)):
                can_pts_py3d_batch, can_dirs_py3d_batch = WarpCanonical.warp_points(
                    rays_points_world=batches_ray_points[batch_idx],
                    rays_directions_world=batches_ray_dirs[batch_idx],
                    vertices_posed=vertices_posed,
                    Ts=Ts,
                )
                can_pts_py3d = torch.cat([can_pts_py3d, can_pts_py3d_batch], dim=1)
                can_dirs_py3d = torch.cat([can_dirs_py3d, can_dirs_py3d_batch], dim=1)

            return can_pts_py3d, can_dirs_py3d


    @staticmethod
    def warp_points_batched_with_cpu(
            rays_points_world,
            rays_directions_world,
            vertices_posed,
            Ts,
            n_batches: int = 16,
    ):
        """
        Args:
            ray_points: torch.Size([1, 8192, 1, 32, 3])
            ray_directions: torch.Size([1, 8192, 1, 32, 3])
            n_batches: int = 16
        """
        assert rays_points_world.device.type == "cpu", "warp_points_batched_with_cpu only works with cpu tensors"


        batches_ray_points = torch.chunk(rays_points_world, chunks=n_batches, dim=1)
        batches_ray_dirs = torch.chunk(rays_directions_world, chunks=n_batches, dim=1)


        # For each batch, execute the standard forward pass and concatenate
        can_pts_py3d = torch.tensor([], device='cpu')
        can_dirs_py3d = torch.tensor([], device='cpu')

        for batch_idx in range(len(batches_ray_points)):
            rays_pts_chunk = batches_ray_points[batch_idx]
            rays_dirs_chunk = batches_ray_dirs[batch_idx]

            can_pts_py3d_batch, can_dirs_py3d_batch = WarpCanonical.warp_points(
                rays_points_world=rays_pts_chunk.to('cuda'),
                rays_directions_world=rays_dirs_chunk.to('cuda'),
                vertices_posed=vertices_posed.to('cuda'),
                Ts=Ts.to('cuda'),
            )
            can_pts_py3d = torch.cat([
                can_pts_py3d,
                can_pts_py3d_batch.to('cpu')
            ], dim=1)
            can_dirs_py3d = torch.cat([
                can_dirs_py3d,
                can_dirs_py3d_batch.to('cpu')
            ], dim=1)

        return can_pts_py3d, can_dirs_py3d



def warp_points_debug(
        rays_points_world,
        vertices,
        Ts,
):
    """
        rays_points_world torch.Size([2672, 32, 3]) OR  torch.Size([1, 2048, 1, 128, 3])
        vertices torch.Size([1, 778, 3])
        Ts torch.Size([1, 778, 4, 4])
    """
    # return rays_points_world

    orig_rays_points_world = rays_points_world.clone()

    if len(orig_rays_points_world.shape) == 3:
        # TODO how to handle this?
        print('rays_points_world.shape', rays_points_world.shape)
        rays_points_world = rays_points_world.unsqueeze(0)

    orig_shape = rays_points_world.shape
    print('orig_shape', orig_shape)

    rays_points_world = rays_points_world.reshape(1, -1, 3)

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

    # generate 4x4 identity matrix, shape [1, Ts.shape[1], 4, 4]
    # T_inv = torch.eye(4, dtype=Ts.dtype, device=Ts.device)[None].repeat(Ts.shape[1], 1, 1).unsqueeze(0)
    T_inv = torch.inverse(Ts)
    print('T_inv', T_inv.shape)

    # res torch.Size([1, 85504])
    res = res.squeeze(0)
    print('res', res.shape)

    # T_inv torch.Size([1, 778, 4, 4])
    # index T_inv with res
    T_inv_vert = T_inv[:, res, :, :] #.squeeze(0)

    print('T_inv_vert', T_inv_vert.shape, T_inv_vert)
    # count how many elements have the same value in res
    res_unique, res_counts = torch.unique(res, return_counts=True)
    print('distances', distances)
    # print('res_unique', res_unique, res_counts)
    print('T_inv_vert', T_inv_vert.shape)

    # convert the points from pytorch3d space to xyz space
    ray_points_xyz = torch.stack(
        (-rays_points_world[..., 0], -rays_points_world[..., 1], rays_points_world[..., 2]),
        -1
    )
    print('ray_points_xyz', ray_points_xyz.shape, ray_points_xyz)

    # add homogeneous coordinate
    ray_points_xyz_homo = torch.cat([
        ray_points_xyz,
        torch.ones_like(ray_points_xyz[..., 0:1])
    ], axis=-1)
    print('ray_points_xyz_homo', ray_points_xyz_homo.shape, ray_points_xyz_homo)

    # Ts torch.Size([1, 342016, 4, 4])
    # ray_points_xyz torch.Size([1, 342016, 3])

    # for each batch, perform torch.bmm. Unsqueeze from [4] to [4, 1] vector
    ray_points_xyz_homo = ray_points_xyz_homo.unsqueeze(-1)
    can_pts_xyz = torch.stack([torch.bmm(T_inv_vert[i], ray_points_xyz_homo[i]) for i in range(T_inv_vert.shape[0])], dim=0)
    print('can_pts_xyz', can_pts_xyz.shape, can_pts_xyz)



    # Squeeze from [4, 1] to [4] vector, drop homogeneous dimension
    can_pts_xyz = can_pts_xyz.squeeze(-1)[..., :3]
    print('can_pts_xyz', can_pts_xyz.shape, can_pts_xyz)

    # can_pts_xyz = (Ts @ ray_points_xyz_homo)  #[:, :3, 0].reshape(human_b, human_n, 3)
    # print(can_pts_xyz)
    # [B x N x 3]
    # get canonical directions
    # can_dirs_xyz = can_pts_xyz[..., 1:] - can_pts_xyz[..., :-1]
    # can_dirs_xyz = torch.cat([can_dirs_xyz, can_dirs_xyz[..., -1:]], dim=-1)
    # can_dirs_xyz = can_dirs_xyz / torch.norm(can_dirs_xyz, dim=-1, keepdim=True)

    # reshape can_pts to the input dimensions [B x N x 3] -> [B x N_rays x N_samples x 3]
    can_pts_xyz = can_pts_xyz.reshape(orig_shape)
    print('can_pts_xyz', can_pts_xyz.shape)

    # can_dirs_xyz = can_pts_xyz[:, :, 1:] - can_pts_xyz[:, :, :-1]
    # can_dirs_xyz = torch.cat([can_dirs_xyz, can_dirs_xyz[:, :, -1:]], dim=2)
    # can_dirs_xyz = can_dirs_xyz / torch.norm(can_dirs_xyz, dim=3, keepdim=True)

    can_dirs_xyz = can_pts_xyz[..., 1:, :] - can_pts_xyz[..., :-1, :]
    can_dirs_xyz = torch.cat([can_dirs_xyz, can_dirs_xyz[..., -1:, :]], dim=-2)
    can_dirs_xyz = can_dirs_xyz / torch.norm(can_dirs_xyz, dim=-1, keepdim=True)


    print('can_dirs_xyz', can_dirs_xyz.shape)


    can_pts_py3d = torch.stack(
        (-can_pts_xyz[..., 0], -can_pts_xyz[..., 1], can_pts_xyz[..., 2]),
        -1
    )
    can_dirs_py3d = torch.stack(
        (-can_dirs_xyz[..., 0], -can_dirs_xyz[..., 1], can_dirs_xyz[..., 2]),
        -1
    )

    # reshape can_pts and can_dirs to the input dimensions
    if len(orig_rays_points_world.shape) == 3:
        # TODO how to handle this?
        can_pts_py3d = can_pts_py3d.squeeze(0)
        can_dirs_py3d = can_dirs_py3d.squeeze(0)
        print('can_pts_py3d.shape', can_pts_py3d.shape)





    ##################################################################################
    # Plots
    ##################################################################################

    import mano_pytorch3d
    hand_model = mano_pytorch3d.MANOCustom(
        model_path='/home/azhuavlev/Desktop/Data/models/mano/MANO_LEFT.pkl',
        is_rhand=False,
        use_pca=False,
    )

    zero_pose_verts = hand_model.forward_pytorch3d(
        betas=torch.zeros(1, 10),
        global_orient=torch.zeros(1, 3),
        hand_pose=torch.zeros(1, 45),
        transl=torch.zeros(1, 3)
    ).reshape(-1, 3)


    # plot the warped points on one plot, and warped directions on the second subplot
    import matplotlib.pyplot as plt

    can_pts_py3d = can_pts_py3d.cpu().numpy().reshape(-1, 3)
    can_dirs_py3d = can_dirs_py3d.cpu().numpy().reshape(-1, 3)
    plot_every_ray = 100

    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(can_pts_py3d[::plot_every_ray, 0], can_pts_py3d[::plot_every_ray, 1], can_pts_py3d[::plot_every_ray, 2], c='r', marker='o', s=0.1)
    ax.scatter(zero_pose_verts[:, 0], zero_pose_verts[:, 1], zero_pose_verts[:, 2], c='b', marker='o', s=1)
    ax.set_title('can_pts_py3d')

    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(can_dirs_py3d[::plot_every_ray, 0], can_dirs_py3d[::plot_every_ray, 1], can_dirs_py3d[::plot_every_ray, 2], c='b', marker='o', s=0.1)
    ax.set_title('can_dirs_py3d')

    # plot ray_points_world
    ray_points_world = orig_rays_points_world.cpu().numpy().reshape(-1, 3)
    ax = fig.add_subplot(133, projection='3d')
    ax.scatter(ray_points_world[::plot_every_ray, 0], ray_points_world[::plot_every_ray, 1], ray_points_world[::plot_every_ray, 2], c='g', marker='o', s=0.1)
    ax.set_title('ray_points_world')

    # plot vertices
    vertices = vertices.cpu().numpy().reshape(-1, 3)
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='y', marker='o', s=1)

    # save plot as pickle
    import pickle
    with open('warped_points.pickle', 'wb') as f:
        pickle.dump(fig, f)

    ##############################################################################################
    import igl
    from utils import utils

    _, uvs, faces = utils.read_obj(
        '/itet-stor/azhuavlev/net_scratch/Projects/Data/models/mano/uv_maps/MANO_UV_left.obj'
    )
    signed_dist, f_id, closest = igl.signed_distance(ray_points_world, vertices, faces[:, :3])
    # how many points are inside the mesh
    inside_world = (signed_dist < 0).sum()

    signed_dist, f_id, closest = igl.signed_distance(can_pts_py3d, zero_pose_verts.cpu().numpy(), faces[:, :3])
    inside_canonical = (signed_dist < 0).sum()

    print('inside_world', inside_world, 'inside_canonical', inside_canonical, f'fraction: {inside_canonical / inside_world:.2f}')


    exit()

    return can_pts_py3d, can_dirs_py3d