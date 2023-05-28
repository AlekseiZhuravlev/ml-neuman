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
import numpy as np


class MANOCustom(smplx.MANO):
    def __init__(self, device, **kwargs):
        super(MANOCustom, self).__init__(**kwargs)
        self.device = device

    def verts_transformations(
            self,
            betas: Optional[Tensor] = None,
            global_orient: Optional[Tensor] = None,
            hand_pose: Optional[Tensor] = None,
            transl: Optional[Tensor] = None,
            return_verts: bool = True,
            return_full_pose: bool = False,
            return_tensor=True,
            concat_joints=False,
            **kwargs):
        """
        Calculate 4x4 affine transformation matrices for each vertex, zero pose -> scene pose
        """

        # render the scene pose, get vertices and joints
        output = self(global_orient=global_orient, hand_pose=hand_pose, betas=betas, transl=transl)
        scene_pose_verts = output.vertices
        scene_pose_joints = output.joints

        # render the zero pose, get vertices and joints
        output = self(
            global_orient=torch.zeros_like(global_orient),
            hand_pose=torch.zeros_like(hand_pose),
            betas=betas,  # torch.zeros_like(shape),
            transl=torch.zeros_like(transl)
        )
        zero_pose_verts, zero_pose_joints = output.vertices, output.joints

        # concatenate vertices and joints, convert to homogeneous coordinates (add 1 to the right)
        scene_verts_joints = ray_utils.to_homogeneous(
            np.concatenate([scene_pose_verts[0], scene_pose_joints[0]], axis=0))
        zero_pose_verts_joints = ray_utils.to_homogeneous(
            np.concatenate([zero_pose_verts[0], zero_pose_joints[0]], axis=0))

        T_t2pose = []
        # calculate 3x4 matrix, then add 1 to the right bottom and zeros
        for i in range(scene_verts_joints.shape[0]):
            # get vertices+joints array of scene pose, drop the last 1, transpose to get 3x1 vector
            scene_params = scene_verts_joints[i][:3][None].T

            # get vertices+joints array of zero pose, transpose to get 4x1 vector
            zero_pose_params = zero_pose_verts_joints[i][None].T

            # calculate 3x4 transformation matrix
            T = scene_params.dot(np.linalg.pinv(zero_pose_params))

            # convert to 4x4 matrix with the last row [0, 0, 0, 1]
            T_homo = np.eye(4)
            T_homo[:3, :4] = T

            T_t2pose.append(T_homo)

        T_t2pose = np.array(T_t2pose)

        if not return_tensor:
            scene_pose_verts = scene_pose_verts.detach().cpu().numpy()[0]
        else:
            T_t2pose = torch.tensor(T_t2pose)

        return scene_pose_verts, T_t2pose

    # this does not work for mano

    # def verts_transformations(
    #         self,
    #         betas: Optional[Tensor] = None,
    #         global_orient: Optional[Tensor] = None,
    #         hand_pose: Optional[Tensor] = None,
    #         transl: Optional[Tensor] = None,
    #         return_verts: bool = True,
    #         return_full_pose: bool = False,
    #         return_tensor=True,
    #         concat_joints=False,
    #         **kwargs
    # ) -> MANOOutput:
    #     ''' Forward pass for the MANO model
    #     '''
    #     # If no shape and pose parameters are passed along, then use the
    #     # ones from the module
    #     global_orient = (global_orient if global_orient is not None else
    #                      self.global_orient)
    #     betas = betas if betas is not None else self.betas
    #     hand_pose = (hand_pose if hand_pose is not None else
    #                  self.hand_pose)
    #
    #     apply_trans = transl is not None or hasattr(self, 'transl')
    #     if transl is None:
    #         if hasattr(self, 'transl'):
    #             transl = self.transl
    #
    #     if self.use_pca:
    #         hand_pose = torch.einsum(
    #             'bi,ij->bj', [hand_pose, self.hand_components])
    #
    #     full_pose = torch.cat([global_orient, hand_pose], dim=1)
    #     full_pose += self.pose_mean
    #
    #     L, vertices = lbs_custom(betas, full_pose, self.v_template,
    #                       self.shapedirs, self.posedirs,
    #                       self.J_regressor, self.parents,
    #                       self.lbs_weights, dtype=self.dtype,
    #                       return_T=True, concat_joints=concat_joints)
    #
    #     # new code
    #     if transl is not None:
    #         transl_4x4 = torch.eye(4, dtype=self.dtype, device=self.device)[None]
    #         transl_4x4[0, :3, 3] = transl.unsqueeze(1)
    #         T = torch.matmul(transl_4x4, L)
    #     else:
    #         T = L
    #     if not return_tensor:
    #         vertices = vertices.detach().cpu().numpy()[0]
    #         T = T.detach().cpu().numpy()[0]
    #     return vertices, T

    # Original code

    # if self.joint_mapper is not None:
    #     joints = self.joint_mapper(joints)
    #
    # if apply_trans:
    #     joints = joints + transl.unsqueeze(dim=1)
    #     vertices = vertices + transl.unsqueeze(dim=1)
    #
    # output = MANOOutput(vertices=vertices if return_verts else None,
    #                     joints=joints if return_verts else None,
    #                     betas=betas,
    #                     global_orient=global_orient,
    #                     hand_pose=hand_pose,
    #                     full_pose=full_pose if return_full_pose else None)
    #
    # return output
