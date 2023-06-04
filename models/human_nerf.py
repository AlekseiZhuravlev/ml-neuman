#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import os
import copy

import torch
import torch.nn as nn

from utils import utils, ray_utils
from models import vanilla
from models.mano import MANOCustom
from typing import Optional, Dict, Union

'''
Extra offset network to compensate the misalignment
'''


class HumanNeRF(nn.Module):
    def __init__(self, opt, poses=None, betas=None, trans=None, scale=None):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.coarse_bkg_net, self.fine_bkg_net = vanilla.build_nerf(opt)
        self.offset_nets = nn.ModuleList([vanilla.build_offset_net(opt) for i in range(opt.num_offset_nets)]).to(
            device
        )

        # canonical space always use 0 as minimum frequency
        temp_opt = copy.deepcopy(opt)
        temp_opt.pos_min_freq = 0
        temp_opt.use_viewdirs = temp_opt.specular_can
        temp_opt.posenc = temp_opt.can_posenc

        self.coarse_human_net, _ = vanilla.build_nerf(temp_opt)
        self.coarse_human_net = self.coarse_human_net.to(device)

        if poses is not None:
            assert betas is not None
            assert scale is not None
            self.poses = torch.nn.parameter.Parameter(torch.from_numpy(poses).float().to(device),
                                                      requires_grad=True)
            self.betas = torch.nn.parameter.Parameter(torch.from_numpy(betas).float().to(device),
                                                      requires_grad=True)
            self.trans = torch.nn.parameter.Parameter(torch.from_numpy(trans).float().to(device),
                                                      requires_grad=True)
            self.scale = scale

            self.hand_model = MANOCustom(
                model_path='/home/azhuavlev/Desktop/Data/models/mano/MANO_LEFT.pkl',
                is_rhand=False,
                device=device,
                use_pca=False,
            ).to(device)

            self.zero_pose = torch.zeros_like(self.poses[0])
            self.zero_pose = torch.nn.parameter.Parameter(self.zero_pose.reshape(1, -1).to(device), requires_grad=False)

            self.poses_orig = poses.copy()
            self.betas_orig = betas.copy()


        # if opt.use_cuda:
        #     # self.coarse_bkg_net = self.coarse_bkg_net.cuda()
        #     # self.fine_bkg_net = self.fine_bkg_net.cuda()
        #     self.offset_nets = self.offset_nets.cuda()
        #     self.coarse_human_net = self.coarse_human_net.cuda()
        #     if poses is not None:
        #         self.poses = torch.nn.Parameter(torch.tensor(poses, device='cuda').float(), requires_grad=True)
        #         self.betas = torch.nn.Parameter(torch.tensor(betas, device='cuda').float(), requires_grad=True)
        #         self.trans = torch.nn.Parameter(torch.tensor(trans, device='cuda').float(), requires_grad=True)
        #
        #         self.hand_model = MANOCustom(
        #             model_path='/home/azhuavlev/Desktop/Data/models/mano/MANO_LEFT.pkl',
        #             is_rhand=False,
        #             device='cuda',
        #             use_pca=False,
        #         ).to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        #         self.zero_pose = torch.nn.Parameter(
        #             torch.tensor(self.zero_pose.detach().numpy(), device='cuda').float(),
        #             requires_grad=False
        #         )


        # try to load pretrained background model
        try:
            raise Exception('not implemented')
            pretrained_bkg = os.path.join(opt.out_dir, opt.load_background, 'checkpoint.pth.tar')
            bkg_weights = torch.load(pretrained_bkg, map_location='cpu')
            utils.safe_load_weights(self.coarse_bkg_net, bkg_weights['coarse_model_state_dict'])
            utils.safe_load_weights(self.fine_bkg_net, bkg_weights['fine_model_state_dict'])
            print(f'pretrained background model loaded from {pretrained_bkg}')
        except Exception as e:
            print(e)
            print('train from scratch')

        # try to load pretrained canonical human model
        try:
            raise Exception('not implemented')
            pretrained_can = os.path.join(opt.out_dir, opt.load_can, 'checkpoint.pth.tar')
            can_weights = torch.load(pretrained_can, map_location='cpu')
            _can_weights = {}
            for k in can_weights['hybrid_model_state_dict'].keys():
                if 'coarse_human_net.' in k:
                    _can_weights[k.split('coarse_human_net.', 1)[1]] = can_weights['hybrid_model_state_dict'][k]
            utils.safe_load_weights(self.coarse_human_net, _can_weights)
            print(f'pretrained canonical human model loaded from {pretrained_can}')
        except Exception as e:
            print(e)
            print('train from scratch')


    def vertex_forward(self,
                       idx,
                       pose: Optional[torch.Tensor] = None,
                       beta: Optional[torch.Tensor] = None,
                       trans: Optional[torch.Tensor] = None
                       ):
        """
        Calls ManoCustom.verts_transformations, but returns unsqueezed T_t2pose
        T_t2pose.shape torch.Size([1, 794, 4, 4]) <class 'torch.Tensor'>
        scene_pose_verts.shape torch.Size([1, 778, 3]) <class 'torch.Tensor'>

        all parameters are on cuda
        """
        if pose is None:
            pose = self.poses[idx][None]
        if beta is None:
            beta = self.betas[idx][None]
        if trans is None:
            trans = self.trans[idx][None]

        # TODO add trans as parameter

        # assert that pose, beta, trans are torch tensors
        assert isinstance(pose, torch.Tensor)
        assert isinstance(beta, torch.Tensor)
        assert isinstance(trans, torch.Tensor)

        # get mano pose and reshape it
        mano_pose = pose.view(-1, 3)

        # root pose is at position 0, pose of rest of the hand is at positions [1:]
        root_pose = mano_pose[0].view(1, 3)
        hand_pose = mano_pose[1:, :].view(1, -1)

        # get betas (called shapes here) and translation vector
        shape = beta.view(1, -1)
        trans = trans.view(1, 3)

        # render the hand in scene pose, get vertices and joints
        output = self.hand_model(global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
        scene_pose_verts = output.vertices
        scene_pose_joints = output.joints

        # # render zero pose, get vertices and joints of the zero pose
        # output = self.hand_model(
        #     global_orient=torch.zeros_like(root_pose),
        #     hand_pose=torch.zeros_like(hand_pose),
        #     betas=shape,  # torch.zeros_like(shape),
        #     transl=torch.zeros_like(trans)
        # )
        # zero_pose_verts, zero_pose_joints = output.vertices, output.joints

        # get transformation matrices from zero pose to scene pose
        _, T_t2pose = self.hand_model.verts_transformations(global_orient=root_pose, hand_pose=hand_pose, betas=shape,
                                                       transl=trans)
        T_t2pose = T_t2pose.unsqueeze(0)


        # print('mano_pose', mano_pose.device)
        # print('root_pose', root_pose.device)
        # print('hand_pose', hand_pose.device)
        # print('shape', shape.device)
        # print('trans', trans.device)
        #
        # print('scene_pose_verts', scene_pose_verts.device)
        # print('scene_pose_joints', scene_pose_joints.device)
        # print('T_t2pose', T_t2pose.device)
        # exit()


        # print('T_t2pose.shape', T_t2pose.shape, type(T_t2pose))
        # print('scene_pose_verts.shape', scene_pose_verts.shape, type(scene_pose_verts))
        # exit()
        return scene_pose_verts, T_t2pose
