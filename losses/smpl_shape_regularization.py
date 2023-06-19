import torch
import pytorch3d
from pytorch3d.loss import point_mesh_face_distance

from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from torch.autograd import Function
from torch.autograd.function import once_differentiable
# import kaolin
from mpl_toolkits.mplot3d import Axes3D, art3d

import torch.nn.functional as F

import igl

# from kaolin.ops.mesh import check_sign

def smpl_shape_regularization(
        hand_model,
        pts,
        dirs,
        pred,
        opt,
):
    # verts = hand_model.zero_pose_vertices
    # faces = hand_model.faces_mano
    #
    # inside_volume = check_sign(verts, faces, pts)
    # print('inside_volume', inside_volume.shape, inside_volume.dtype)
    # print('kaolin_sum', sum(inside_volume.int()), inside_volume.shape[0] * inside_volume.shape[1])
    #
    # exit()



    smpl_reg_inside = torch.tensor(0.0, requires_grad=True, device=pts.device)
    smpl_reg_outside = torch.tensor(0.0, requires_grad=True, device=pts.device)

    # calculate the distance of each point to the samples inside the mesh
    dist_to_inside_points = torch.cdist(
        pts,
        hand_model.zero_pose_inside_samples
    )

    # for each point, get the indices of 5 closest inside points
    closest_inside_points_dist, closest_inside_points_ind = torch.topk(dist_to_inside_points, 20, dim=2, largest=False)
    min_dist, min_ind = torch.min(closest_inside_points_dist, dim=2)

    # for each inside point, get the distance to surface
    closest_inside_points_dist_to_surface = torch.gather(
        hand_model.distances_samples_surface,
        0,
        closest_inside_points_ind.reshape(-1)
    ).reshape(closest_inside_points_ind.shape)

    # if the distance to one of the 5 closest inside points is smaller than the distance
    # of this inside point to the surface, then the point is inside the mesh
    inside_volume = torch.any(closest_inside_points_dist < closest_inside_points_dist_to_surface, dim=2)

    # inside_volume torch.Size([4096, 32]) torch.bool
    # pred torch.Size([4096, 32, 4]) torch.float32
    # min_dist torch.Size([4096, 32]) torch.float32
    # pred[inside_volume] torch.Size([15470, 4])

    n_points_inside = inside_volume.sum()

    # print('pred.shape', pred.shape)
    # print('inside_volume.shape', inside_volume.shape)
    # print('n_points_inside', n_points_inside)
    # print('pred[inside_volume].shape', pred[inside_volume].shape)

    # penalize the points inside the human mesh
    if n_points_inside > 0:
        # mse error with ones
        smpl_reg_inside = smpl_reg_inside + F.mse_loss(
            1 - torch.exp(-torch.relu(pred[inside_volume][:, 3])),
            torch.ones_like(pred[inside_volume][:,3])
        ) * opt.penalize_smpl_alpha

    # penalize the points outside the human mesh
    if n_points_inside < inside_volume.shape[0] * inside_volume.shape[1]:
        # mse error with zeros, alpha is multiplied with distance to surface ** dist_exponent
        smpl_reg_outside = smpl_reg_outside + F.l1_loss(
            (1 - torch.exp(-torch.relu(pred[~inside_volume][:, 3])))
            * torch.pow(
                torch.abs(min_dist[~inside_volume]) * opt.penalize_outside_factor,
                opt.dist_exponent
            ),
            torch.zeros_like(pred[~inside_volume][:, 3])
        ) * opt.penalize_dummy


    return {
        'smpl_reg_inside': smpl_reg_inside,
        'smpl_reg_outside': smpl_reg_outside,
        'n_points_inside': n_points_inside,
    }




def smpl_shape_regularization_debug(
        hand_model,
        pts,
        dirs,
        pred,
        shape_reg_amount,
):
    """
    To avoid blobs in the canonical space and semi-transparent canonical human,
    we enforce the volume inside the canonical SMPL mesh to be solid,
    while enforcing the volume outside the canonical SMPL mesh to be empty
    """

    smpl_reg = torch.tensor(0.0, requires_grad=True, device=pts.device)

    print('verts', hand_model.zero_pose_vertices.shape, hand_model.zero_pose_vertices)
    # print('faces', faces.shape, faces)
    print('pts', pts.shape, pts)

    dist_to_zero_verts = torch.cdist(
        pts,
        hand_model.zero_pose_vertices
    )
    dist_to_inside_points = torch.cdist(
        pts,
        hand_model.zero_pose_inside_samples
    )

    print('dist_to_zero_verts', dist_to_zero_verts.shape, dist_to_zero_verts)
    print('dist_to_inside_points', dist_to_inside_points.shape, dist_to_inside_points)

    closest_verts = torch.argmin(dist_to_zero_verts, dim=2)
    closest_inside_points = torch.argmin(dist_to_inside_points, dim=2)

    print('closest_verts', closest_verts.shape, closest_verts)
    print('closest_inside_points', closest_inside_points.shape, closest_inside_points)

    dist_to_closest_verts = torch.gather(dist_to_zero_verts, 2, closest_verts.unsqueeze(2)).squeeze(2)
    dist_to_closest_inside_points = torch.gather(dist_to_inside_points, 2, closest_inside_points.unsqueeze(2)).squeeze(2)

    # dist_to_closest_verts = dist_to_zero_verts[closest_verts]
    # dist_to_closest_inside_points = dist_to_inside_points[closest_inside_points]
    dist_btw_closest_inside_points_and_verts = hand_model.distances_samples_vertices[closest_inside_points, closest_verts]

    print('dist_to_closest_verts', dist_to_closest_verts.shape, dist_to_closest_verts)
    print('dist_to_closest_inside_points', dist_to_closest_inside_points.shape, dist_to_closest_inside_points)
    print('dist_btw_closest_inside_points_and_verts', dist_btw_closest_inside_points_and_verts.shape, dist_btw_closest_inside_points_and_verts)

    inside_volume = dist_to_closest_inside_points < dist_btw_closest_inside_points_and_verts

    print('inside_volume', inside_volume.shape, inside_volume)

    # print how many points are inside the mesh
    print('sum', torch.sum(inside_volume.int()), inside_volume.shape[0] * inside_volume.shape[1])


    print('hand_model.distances_samples_surface', hand_model.distances_samples_surface.shape, hand_model.distances_samples_surface)

    # for each point, get the indices of 5 closest inside points
    closest_inside_points_dist, closest_inside_points_ind = torch.topk(dist_to_inside_points, 20, dim=2, largest=False)
    # print('closest_inside_points', closest_inside_points_dist.shape, closest_inside_points_dist)

    # for each inside point, get the distance to surface
    # closest_inside_points_dist_to_surface = hand_model.distances_samples_surface[closest_inside_points_ind]
    # rewrite the above line to avoid indexing with a tensor
    # print(torch.gather(hand_model.distances_samples_surface, 0, closest_inside_points_ind).shape)
    closest_inside_points_dist_to_surface = torch.gather(
        hand_model.distances_samples_surface,
        0,
        closest_inside_points_ind.reshape(-1)
    ).reshape(closest_inside_points_ind.shape)

    print('closest_inside_points_dist_to_surface', closest_inside_points_dist_to_surface.shape)
    # exit()

    # if the distance to one of the 5 closest inside points is smaller than the distance
    # of this inside point to the surface, then the point is inside the mesh
    inside_volume = torch.any(closest_inside_points_dist < closest_inside_points_dist_to_surface, dim=2)

    print('inside_volume', inside_volume.shape, inside_volume)
    print('sum - after warping, custom', torch.sum(inside_volume.int()), inside_volume.shape[0] * inside_volume.shape[1])

    # exit()





    # make a 3d plot of points, each point color coded by whether it is inside or outside the mesh
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')

    n_batch = 100
    pts = pts.reshape(-1, 3).detach().cpu().numpy()#[1024 * n_batch:1024 * (n_batch+3), :]
    inside_volume = inside_volume.reshape(-1).detach().cpu().numpy()#[1024 * n_batch:1024 * (n_batch+3)]


    verts = hand_model.zero_pose_vertices.detach().cpu().numpy()
    pts_inside = hand_model.zero_pose_inside_samples.detach().cpu().numpy()

    dist_human, _, _ = igl.signed_distance(
        pts,
        verts,
        hand_model.faces_mano.cpu().numpy(),
    )

    ax.set_title('sampling method')
    # inside_volume = dist_human < 0

    # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=~inside_volume, s=0.7, cmap='bwr')

    # only display points that are inside the mesh
    ax.scatter(pts[inside_volume, 0], pts[inside_volume, 1], pts[inside_volume, 2], c='red', s=0.7)
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='black', s=0.7)
    # ax.scatter(pts_inside[:, 0], pts_inside[:, 1], pts_inside[:, 2], c='red', s=1)

    ax = fig.add_subplot(122, projection='3d')
    ax.set_title('igl method')

    import numpy as np
    inside_volume = dist_human < 0
    print(inside_volume)
    print('sum - after warping, igl', sum(inside_volume), len(inside_volume))


    # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=~inside_volume, s=0.7, cmap='bwr')
    # only display points that are inside the mesh
    ax.scatter(pts[inside_volume, 0], pts[inside_volume, 1], pts[inside_volume, 2], c='red', s=0.7)
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='black', s=0.7)


    import pickle
    with open('/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/inside_volume.pkl', 'wb') as f:
        pickle.dump(fig, f)

    # plt.show()

    exit()

    # todo SMPL shape loss: we need to penalize alpha of points far from the mesh

    res = torch.cdist(pts.reshape(1, -1, 3), verts)
    res = torch.min(res, dim=1)[0]

    smpl_reg = smpl_reg + torch.mean(res) * shape_reg_amount

    return smpl_reg



def _smpl_shape_regularization_original(
        self,
        batch,
        pts,
        dirs,
        pred
):
    """
    To avoid blobs in the canonical space and semi-transparent canonical human,
    we enforce the volume inside the canonical SMPL mesh to be solid,
    while enforcing the volume outside the canonical SMPL mesh to be empty
    """



    smpl_reg = torch.tensor(0.0, requires_grad=True, device=pts.device)

    # get the canonical mesh
    can_mesh = self.val_dataset.scene.captures[batch['cap_id']].can_mesh

    # TODO change to torch

    # calculate the distance from canonical points to the human mesh
    dist_human, _, _ = igl.signed_distance(
        pts.reshape(-1, 3).detach().cpu().numpy(),
        can_mesh.verts_packed().cpu().numpy(),
        can_mesh.faces_packed().cpu().numpy(),
    )

    # get the points inside the human mesh
    inside_volume = dist_human < 0

    # penalize the points inside the human mesh
    if inside_volume.sum() > 0:
        smpl_reg = smpl_reg + F.mse_loss(
            1 - torch.exp(-torch.relu(pred.reshape(-1, 4)[inside_volume][:, 3])),
            torch.ones_like(pred.reshape(-1, 4)[inside_volume][:, 3])
        ) * self.penalize_smpl_alpha


    if self.penalize_dummy > 0:
        # generate random samples inside a box in canonical space
        dummy_pts = (torch.rand(pts.shape, dtype=pts.dtype, device=device) - 0.5) * 3
        dummy_out = self.model.coarse_human_net(dummy_pts, dirs)

        # TODO change to torch
        # calculate the distance from canonical points to the human mesh
        dist_dummy, _, _ = igl.signed_distance(
            dummy_pts.reshape(-1, 3).detach().cpu().numpy(),
            can_mesh.verts_packed().cpu().numpy(),
            can_mesh.faces_packed().cpu().numpy(),
        )

        # get the points inside the human mesh
        dummy_inside = dist_dummy < 0
        dummy_outside = dist_dummy > 0

        # penalize the points inside the human mesh
        if dummy_inside.sum() > 0:
            smpl_reg = smpl_reg + F.mse_loss(
                1 - torch.exp(-torch.relu(dummy_out.reshape(-1, 4)[dummy_inside][:, 3])),
                torch.ones_like(dummy_out.reshape(-1, 4)[dummy_inside][:, 3])
            ) * self.penalize_dummy

        # penalize the points outside the human mesh
        if dummy_outside.sum() > 0:
            smpl_reg = smpl_reg + F.l1_loss(
                (1 - torch.exp(-torch.relu(dummy_out.reshape(-1, 4)[dummy_outside][:, 3])))
                * torch.pow(
                    torch.abs(torch.from_numpy(dist_dummy[dummy_outside]).to(device)) * self.opt.penalize_outside_factor,
                    self.opt.dist_exponent
                ),
                torch.zeros_like(dummy_out.reshape(-1, 4)[dummy_outside][:, 3])
            ) * self.penalize_dummy

    return smpl_reg