import smplx
import torch
from smplx.utils import (
    Struct, to_np, to_tensor, Tensor, Array,
    SMPLOutput,
    SMPLHOutput,
    SMPLXOutput,
    MANOOutput,
    FLAMEOutput)
from typing import Optional, Dict, Union


from mano_custom.smpl_custom import lbs as lbs_custom


def create_mano_custom(
        return_right_hand: bool,
):
    # keep flat_hand_mean=False, in images it is the only way.
    # Confirmed by running dataloader_project_mano.py
    left_hand = _MANOCustom(
        model_path='/home/azhuavlev/Desktop/Data/models/mano/MANO_LEFT.pkl',
        is_rhand=False,
        use_pca=False,
        flat_hand_mean=False
    )
    right_hand = _MANOCustom(
        model_path='/home/azhuavlev/Desktop/Data/models/mano/MANO_RIGHT.pkl',
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=False
    )
    if torch.sum(torch.abs(left_hand.shapedirs[:, 0, :] - right_hand.shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        left_hand.shapedirs[:, 0, :] *= -1
    else:
        print('Checked MANO, no need to change shapedirs')

    if return_right_hand:
        return right_hand
    else:
        return left_hand



class _MANOCustom(smplx.MANO):
    def __init__(self, **kwargs):
        super(_MANOCustom, self).__init__(**kwargs)

    def verts_transformations_xyz(
            self,
            betas,
            global_orient,
            hand_pose,
            transl,
            **kwargs
    ) -> MANOOutput:
        ''' Forward pass for the MANO model
        Same as unbatched, but will not squeeze the batch dimension of T and vertices
        returns
        vertices torch.Size([1, 778, 3]) - ZERO POSE vertices in XYZ format !!!
        T torch.Size([1, 778, 4, 4]) - !!! transformation matrices in XYZ format !!!
        '''

        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        full_pose += self.pose_mean

        # print('pose_mean', self.pose_mean)

        L, vertices = lbs_custom(betas, full_pose, self.v_template,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype,
                          return_T=True, concat_joints=False)

        # new code
        if transl is not None:
            transl_4x4 = torch.eye(4, dtype=self.dtype,
                                   device=L.device
                                   )[None]
            transl_4x4[0, :3, 3] = transl.unsqueeze(1)
            T = torch.matmul(transl_4x4, L)
        else:
            T = L

        # vertices = torch.stack((-vertices[:, :, 0], -vertices[:, :, 1], vertices[:, :, 2]),
        #                     2)
        return vertices, T

    def forward_pytorch3d(self,
                          betas,
                          global_orient,
                          hand_pose,
                          transl
                          ):
        """
        Same as forward, but returns vertices in PyTorch3D format
        """
        output =  self(
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose,
            transl=transl,
        )
        verts = output.vertices

        # reverse x- and y-axis following PyTorch3D axis direction
        return torch.stack((-verts[:, :, 0], -verts[:, :, 1], verts[:, :, 2]),
                            2)

    def get_flat_hand_vertices_pytorch3d(self, device):
        zero_pose_hand = self.forward_pytorch3d(
            betas = torch.zeros(1, 10, device=device),
            global_orient = -self.pose_mean[:3].unsqueeze(0),
            hand_pose = -self.pose_mean[3:].unsqueeze(0),
            transl = torch.zeros(1, 3, device=device),
        )
        return zero_pose_hand