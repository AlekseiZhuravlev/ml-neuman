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
    def __init__(self, **kwargs):
        super(MANOCustom, self).__init__(flat_hand_mean=True, **kwargs)
        # self.device = device

    # this is wrong
    # def verts_transformations(
    #         self,
    #         betas,
    #         global_orient,
    #         hand_pose,
    #         transl,
    #         return_verts: bool = True,
    #         return_full_pose: bool = False,
    #         return_tensor=True,
    #         concat_joints=False,
    #         **kwargs):
    #     """
    #     Calculate 4x4 affine transformation matrices for each vertex, zero pose -> scene pose
    #     """
    #     # TODO move everything to cuda
    #
    #     # render the scene pose, get vertices and joints
    #     output = self(global_orient=global_orient, hand_pose=hand_pose, betas=betas, transl=transl)
    #     scene_pose_verts = output.vertices
    #     scene_pose_joints = output.joints
    #
    #     # render the zero pose, get vertices and joints
    #     output = self(
    #         global_orient=torch.zeros_like(global_orient),
    #         hand_pose=torch.zeros_like(hand_pose),
    #         betas=betas,  # torch.zeros_like(shape),
    #         transl=torch.zeros_like(transl)
    #     )
    #     zero_pose_verts, zero_pose_joints = output.vertices, output.joints
    #
    #     # concatenate vertices and joints, convert to homogeneous coordinates (add 1 to the right)
    #     # scene_verts_joints = ray_utils.to_homogeneous(
    #     #     np.concatenate([scene_pose_verts[0], scene_pose_joints[0]], axis=0))
    #     # zero_pose_verts_joints = ray_utils.to_homogeneous(
    #     #     np.concatenate([zero_pose_verts[0], zero_pose_joints[0]], axis=0))
    #     scene_verts_joints = torch.cat([scene_pose_verts[0], scene_pose_joints[0]], axis=0)
    #     zero_pose_verts_joints = ray_utils.to_homogeneous(
    #         torch.cat([zero_pose_verts[0], zero_pose_joints[0]], axis=0))
    #
    #     # get 3x4 transformation matrix from zero pose to scene pose
    #
    #     # print('torch.linalg.pinv(zero_pose_verts_joints)', )
    #     zero_pose_to_scene_pose = scene_verts_joints.unsqueeze(-1) @ (torch.linalg.pinv(zero_pose_verts_joints.unsqueeze(-1)))
    #     # add [0, 0, 0, 1] to the right bottom
    #
    #     # print('zero_pose_to_scene_pose', zero_pose_to_scene_pose.device)
    #     zero_pose_to_scene_pose = torch.cat([
    #         zero_pose_to_scene_pose,
    #         torch.tensor([0, 0, 0, 1], device=zero_pose_to_scene_pose.device
    #                      ).unsqueeze(0).unsqueeze(0).repeat(zero_pose_to_scene_pose.shape[0], 1, 1)
    #     ], dim=1)
    #     # print('zero_pose_to_scene_pose', zero_pose_to_scene_pose.shape)
    #     # exit(0)
    #
    #     # print('scene_pose_verts', scene_pose_verts.device)
    #     # print('zero_pose_verts', zero_pose_verts.device)
    #     # exit(0)
    #
    #     return scene_pose_verts, zero_pose_to_scene_pose

        # T_t2pose = []

        # an inefficient way to get 4x4 transformation matrix from zero pose to scene pose

        # print('scene_verts_joints', scene_verts_joints.shape)
        # print('zero_pose_verts_joints', zero_pose_verts_joints.shape)
        # exit(0)
        # calculate 3x4 matrix, then add 1 to the right bottom and zeros
        # for i in range(scene_verts_joints.shape[0]):
        #     # get vertices+joints array of scene pose, drop the last 1, transpose to get 3x1 vector
        #     scene_params = scene_verts_joints[i][None].T
        #
        #     # get vertices+joints array of zero pose, transpose to get 4x1 vector
        #     zero_pose_params = zero_pose_verts_joints[i][None].T
        #
        #     print('scene_params', scene_params.shape, scene_params)
        #     print('zero_pose_params', zero_pose_params.shape, zero_pose_params)
        #     print('torch.linalg.pinv(zero_pose_params)', torch.linalg.pinv(zero_pose_params).shape, torch.linalg.pinv(zero_pose_params))
        #     # calculate 3x4 transformation matrix
        #     T = scene_params @ (torch.linalg.pinv(zero_pose_params))
        #
        #     # print('T', T.shape, T)
        #     # lstsq = torch.linalg.lstsq(zero_pose_params, scene_params)
        #     # print('lstsq', lstsq.shape, lstsq)
        #
        #     # print('T', T.shape, T)
        #
        #     # convert to 4x4 matrix with the last row [0, 0, 0, 1]
        #     T_homo = torch.eye(4)
        #     T_homo[:3, :4] = T
        #
        #     T_t2pose.append(T_homo)
        #
        #     print('zero_pose_to_scene_pose[i]', zero_pose_to_scene_pose[i])
        #     print('T_t2pose', T_homo)
        #
        # exit(0)
        #
        # # T_t2pose = np.array(T_t2pose)
        # # convert list of tensors to tensor
        # # print('len(T_t2pose)', len(T_t2pose), T_t2pose[0].shape)
        # T_t2pose = torch.stack(T_t2pose)
        # # print('T_t2pose.shape', T_t2pose.shape)
        # # exit(0)
        # if not return_tensor:
        #     scene_pose_verts = scene_pose_verts.detach().cpu().numpy()[0]
        #     T_t2pose = T_t2pose.detach().cpu().numpy()
        # # else:
        # #     T_t2pose = torch.tensor(T_t2pose)
        #
        # return scene_pose_verts, T_t2pose

    # this does not work for mano

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
            **kwargs
    ) -> MANOOutput:
        ''' Forward pass for the MANO model
        returns
        T torch.Size([778, 4, 4])
        vertices torch.Size([778, 3])
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        betas = betas if betas is not None else self.betas
        hand_pose = (hand_pose if hand_pose is not None else
                     self.hand_pose)

        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        # if self.use_pca:
        #     hand_pose = torch.einsum(
        #         'bi,ij->bj', [hand_pose, self.hand_components])

        # print('global_orient', global_orient.shape, global_orient)
        #
        # print('hand_pose', hand_pose.shape, hand_pose)
        # print('betas', betas.shape, betas)
        # exit(0)

        # assert that input variables are torch tensors
        assert isinstance(global_orient, torch.Tensor)
        assert isinstance(betas, torch.Tensor)
        assert isinstance(hand_pose, torch.Tensor)
        if transl is not None:
            assert isinstance(transl, torch.Tensor)


        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        full_pose += self.pose_mean

        L, vertices = lbs_custom(betas, full_pose, self.v_template,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype,
                          return_T=True, concat_joints=concat_joints)

        # new code
        if transl is not None:
            transl_4x4 = torch.eye(4, dtype=self.dtype,
                                   device=L.device
                                   )[None]
            transl_4x4[0, :3, 3] = transl.unsqueeze(1)
            T = torch.matmul(transl_4x4, L)
        else:
            T = L
        if not return_tensor:
            raise TypeError('Trying to return numpy array instead of tensor')
            # vertices = vertices.detach().cpu().numpy()[0]
            # T = T.detach().cpu().numpy()[0]

        T = T.squeeze(0)
        vertices = vertices.squeeze(0)

        # print('T', T.shape)
        # print('vertices', vertices.shape)
        # exit(0)

        # TODO: check neuman code, the dimensions of T and vertices in this function
        return vertices, T


        # test the new code
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

        # get vertices+joints array of zero pose, transpose to get 4x1 vector
        zero_pose_verts = ray_utils.to_homogeneous(zero_pose_verts)
        # get vertices+joints array of scene pose, transpose to get 4x1 vector
        scene_pose_verts = ray_utils.to_homogeneous(scene_pose_verts)

        # get transformation matrix from zero pose to scene pose
        print('zero_pose_verts_joints.shape', zero_pose_verts.shape)
        print('scene_pose_verts_joints.shape', scene_pose_verts.shape)

        print('T.shape', T.shape)

        forward_transf = T @ zero_pose_verts.unsqueeze(-1)
        backward_transf = torch.linalg.inv(T) @ scene_pose_verts.unsqueeze(-1)

        for i in range(T.shape[1]):
            # print(scene_pose_verts[0, i, :], forward_transf[0, i, :].squeeze(-1))
            # print(zero_pose_verts[0, i, :], backward_transf[0, i, :].squeeze(-1))
            #
            # exit()
            assert torch.allclose(scene_pose_verts, forward_transf.squeeze(-1), atol=1e-2), 'T_p @ zero_p != scene_p'
            assert torch.allclose(zero_pose_verts, backward_transf.squeeze(-1), atol=1e-2), 'T_p @ zero_p != scene_p'

        exit(0)

        for i in range(T.shape[1]):
            zero_p = zero_pose_verts[0, i, :]
            scene_p = scene_pose_verts[0, i, :]
            T_p = T[0, i, :, :]
            # print('zero_p', zero_p)
            # print('scene_p', scene_p)
            # print('T_p', T_p)

            # print('T_p @ zero_p', T_p @ zero_p)
            # print('inv(T_p) @ scene_p', torch.linalg.inv(T_p) @ scene_p)
            # verify that the difference between T_p @ zero_p and scene_p is less than 1%
            assert torch.allclose(T_p @ zero_p, scene_p, atol=1e-2), 'T_p @ zero_p != scene_p'
            assert torch.allclose(torch.linalg.inv(T_p) @ scene_p, zero_p, atol=1e-2), 'T_p @ zero_p != scene_p'

            # exit(0)
        exit(0)



        return vertices, T

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
