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

from models.smpl import lbs as lbs_custom
from utils import utils, ray_utils

import trimesh
import igl




import lightning as L


class MANOCustom(smplx.MANO):
    def __init__(self, **kwargs):
        super(MANOCustom, self).__init__(flat_hand_mean=False, **kwargs)


    def verts_transformations_pytorch3d(
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
        vertices torch.Size([1, 778, 3]) - vertices in PyTorch3D format
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

        vertices = torch.stack((-vertices[:, :, 0], -vertices[:, :, 1], vertices[:, :, 2]),
                            2)
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