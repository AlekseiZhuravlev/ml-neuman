import torch
import pytorch3d
from pytorch3d.loss import point_mesh_face_distance

from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import kaolin
from mpl_toolkits.mplot3d import Axes3D, art3d

from kaolin.ops.mesh import check_sign

def smpl_shape_regularization(
        verts,
        faces,
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

    pts = torch.zeros_like(pts)

    print('verts', verts.shape, verts)
    print('faces', faces.shape, faces)
    print('pts', pts.shape, pts)






    inside_volume = kaolin.ops.mesh.check_sign(
        verts,
        faces[0],
        pts.reshape(1, -1, 3)

    )
    print('inside_volume', inside_volume.shape, inside_volume)

    # print how many elements are inside the mesh
    print('inside_volume', inside_volume.int().sum(), inside_volume)

    # display vertices and faces with matplotlib

    # make a 3d plot of points, each point color coded by whether it is inside or outside the mesh
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')



    n_batch = 100
    pts = pts.reshape(-1, 3).detach().cpu().numpy()[1024 * n_batch:1024 * (n_batch+1), :]
    inside_volume = inside_volume[0].detach().cpu().numpy()[1024 * n_batch:1024 * (n_batch+1)]
    faces = faces[0].detach().cpu().numpy()
    verts = verts[0].detach().cpu().numpy()

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=inside_volume)
    #
    # # plot vertices
    # verts = verts.detach().cpu().numpy()[0]

    # display the vertices alongside with their indices
    # for i, v in enumerate(verts):
    #     ax.scatter(v[0], v[1], v[2], c='red')
    #     ax.text(v[0], v[1], v[2], str(i), color='red')




    pc = art3d.Poly3DCollection(verts[faces], edgecolor="black")
    ax.add_collection(pc)

    # save the plot to '/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/inside_volume.pkl'
    import pickle
    with open('/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/inside_volume.pkl', 'wb') as f:
        pickle.dump(fig, f)

    # plt.show()

    exit()

    # todo SMPL shape loss: we need to penalize alpha of points far from the mesh

    res = torch.cdist(pts.reshape(1, -1, 3), verts)
    res = torch.min(res, dim=1)[0]

    smpl_reg = smpl_reg + torch.mean(res) * shape_reg_amount

    # exit()

    return smpl_reg

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
                (1 - torch.exp(-torch.relu(dummy_out.reshape(-1, 4)[dummy_outside][:, 3]))) * torch.pow(torch.abs(
                    torch.from_numpy(dist_dummy[dummy_outside]).to(device)) * self.opt.penalize_outside_factor,
                                                                                                        self.opt.dist_exponent),
                torch.zeros_like(dummy_out.reshape(-1, 4)[dummy_outside][:, 3])
            ) * self.penalize_dummy

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