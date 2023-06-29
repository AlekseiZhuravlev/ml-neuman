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

        self.register_buffer('zero_pose_vertices', None)
        self.register_buffer('zero_pose_inside_samples', None)
        self.register_buffer('distances_samples_vertices', None)
        self.register_buffer('distances_samples_surface', None)

        self.register_buffer('faces_mano', None)

        # self.zero_pose_vertices = None
        # self.zero_pose_inside_samples = None
        # self.distances_samples_vertices = None

        # self.get_zero_pose_inside_samples()

    def get_zero_pose_inside_samples(self):

        _, uvs, faces = utils.read_obj(
            '/itet-stor/azhuavlev/net_scratch/Projects/Data/models/mano/uv_maps/MANO_UV_left.obj'
        )

        self.faces_mano = torch.tensor(faces[:, :3], dtype=torch.int32)

        output = self(
            global_orient=torch.zeros(1,3),
            hand_pose=torch.zeros(1, 45),
            betas=torch.zeros(1, 10),
            transl=torch.zeros(1, 3)
        )
        zero_pose_verts = output.vertices[0]

        mesh = trimesh.Trimesh(vertices=zero_pose_verts, faces=faces[:, :3])

        self.zero_pose_inside_samples = torch.tensor(trimesh.sample.volume_mesh(mesh, 5000), dtype=torch.float32)
        self.zero_pose_vertices = zero_pose_verts

        self.distances_samples_vertices = torch.cdist(
            self.zero_pose_inside_samples,
            self.zero_pose_vertices
        )

        dist_samples_surface, _, _ = igl.signed_distance(
            self.zero_pose_inside_samples.numpy(),
            self.zero_pose_vertices.numpy(),
            faces[:, :3],
        )

        self.distances_samples_surface = torch.tensor(-dist_samples_surface, dtype=torch.float32)

        # print('zero_pose_inside_samples', self.zero_pose_inside_samples.shape, self.zero_pose_inside_samples.dtype)
        # print('zero_pose_vertices', self.zero_pose_vertices.shape, self.zero_pose_vertices.dtype)
        # print('distances_samples_vertices', self.distances_samples_vertices.shape)
        # exit()


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

        return vertices, T

