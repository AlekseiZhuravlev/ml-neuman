#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils, ray_utils, render_utils
from models import vanilla
from models.mano import MANOCustom
from typing import Optional, Dict, Union

from utils.constant import HARD_SURFACE_OFFSET, PATCH_SIZE, PATCH_SIZE_SQUARED, CANONICAL_ZOOM_FACTOR, \
    CANONICAL_CAMERA_DIST


from losses import color_range_regularization, smpl_symmetry_regularization, smpl_shape_regularization, \
    sparsity_regularization, rgb_loss, lpips_loss
import lpips

# import random

import lightning as L

LOSS_NAMES = [
    'fine_rgb_loss',
    'lpips_loss',
    'color_range_reg',
    'smpl_sym_reg',
    'smpl_shape_reg',
    'mask_loss',
    'sparsity_reg'
]

class HumanNeRF(L.LightningModule):
    def __init__(self, opt, poses=None, betas=None, trans=None, scale=None):
        super().__init__()

        # self.coarse_bkg_net, self.fine_bkg_net = vanilla.build_nerf(opt)
        self.offset_nets = nn.ModuleList([vanilla.build_offset_net(opt) for i in range(opt.num_offset_nets)])

        # canonical space always use 0 as minimum frequency
        temp_opt = copy.deepcopy(opt)
        temp_opt.pos_min_freq = 0
        temp_opt.use_viewdirs = temp_opt.specular_can
        temp_opt.posenc = temp_opt.can_posenc
        self.opt = temp_opt

        self.coarse_human_net, _ = vanilla.build_nerf(temp_opt)
        self.coarse_human_net = self.coarse_human_net

        if self.opt.penalize_lpips > 0:
            self.lpips_loss_fn = lpips.LPIPS(net='alex')
            for param in self.lpips_loss_fn.parameters():
                param.requires_grad = False

        if poses is not None:
            assert betas is not None
            assert scale is not None
            self.poses = torch.nn.parameter.Parameter(torch.from_numpy(poses).float(),
                                                      requires_grad=True)
            self.betas = torch.nn.parameter.Parameter(torch.from_numpy(betas).float(),
                                                      requires_grad=True)
            self.trans = torch.nn.parameter.Parameter(torch.from_numpy(trans).float(),
                                                      requires_grad=True)
            self.scale = scale

            self.hand_model = MANOCustom(
                model_path='/home/azhuavlev/Desktop/Data/models/mano/MANO_LEFT.pkl',
                is_rhand=False,
                use_pca=False,
            )
            for param in self.hand_model.parameters():
                param.requires_grad = False

        # try to load pretrained canonical human model
        self.load_pretrained_human_model(opt)


    def load_pretrained_human_model(self, opt):
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


    def forward(self, batch):
        """
        Calculates Fine RGB Map of a trained Human NeRF
        """
        batch = utils.remove_first_axis(batch)

        # get results of human nerf
        _, human_dirs, human_z_vals, can_pts, can_dirs, human_out = self._eval_human_samples(batch)

        # get rgb map
        fine_rgb_map = self.render_rgb_map(
            human_z_vals=human_z_vals,
            human_dirs=human_dirs,
            human_out=human_out,
            white_bkg=self.opt.white_bkg,
        )

        return fine_rgb_map

    def configure_optimizers(self):
        optim_list = [
            {"params": self.poses, "lr": self.opt.smpl_lr},
            {"params": self.coarse_human_net.parameters(), "lr": self.opt.learning_rate},
            {"params": self.offset_nets.parameters(), "lr": self.opt.learning_rate},
        ]
        optimizer = torch.optim.Adam(optim_list)
        return optimizer


    def training_step(self, batch, return_rgb=False):
        """
        Evaluates human nerf on batch, calculates losses
        """
        batch = utils.remove_first_axis(batch)

        # initialize loss dictionary
        loss_dict = {l: torch.tensor(0.0, requires_grad=True, dtype=torch.float32) for l in LOSS_NAMES}

        # get results of human nerf
        _, human_dirs, human_z_vals, can_pts, can_dirs, human_out = self._eval_human_samples(batch)

        # get rgb map
        fine_rgb_map = self.render_rgb_map(
            human_z_vals=human_z_vals,
            human_dirs=human_dirs,
            human_out=human_out,
            white_bkg=self.opt.white_bkg,
        )

        # Calculate losses

        # canonical space should be symmetric in terms of occupancy
        # if self.penalize_symmetric_alpha > 0:
        #     loss_dict['smpl_sym_reg'] = loss_dict['smpl_sym_reg'] + self._smpl_symmetry_regularization(can_pts, can_dirs, human_out)

        # color of the same point should not change too much due to viewing directions
        if self.opt.penalize_color_range > 0:
            loss_dict['color_range_reg'] = loss_dict['color_range_reg'] + \
                                           color_range_regularization.color_range_regularization(
                                               coarse_human_net=self.coarse_human_net,
                                               pts=can_pts,
                                               dirs=can_dirs,
                                               tgts=human_out,
                                               penalize_color_range=self.opt.penalize_color_range
                                           )

        # the rendered human should be close to the detected human mask
        # loosely enforced, the penalty linearly decreases during training
        if self.opt.penalize_mask > 0:
            _, _, human_mask, _, _ = render_utils.raw2outputs(
                human_out,
                human_z_vals,
                human_dirs[:, 0, :],
                white_bkg=self.opt.white_bkg

            )
            loss_dict['mask_loss'] = loss_dict['mask_loss'] + F.mse_loss(
                torch.clamp(human_mask, min=0.0, max=1.0),
                (1.0-batch['is_bkg'].float())
            ) * self.opt.penalize_mask

        # alpha inside smpl mesh should be 1, alpha outside smpl mesh should be 0
        # if self.penalize_smpl_alpha > 0:
        #     loss_dict['smpl_shape_reg'] = loss_dict['smpl_shape_reg'] + self._smpl_shape_regularization(batch, can_pts, can_dirs, human_out)

        # sharp edge loss + hard surface loss
        # USES NUMPY
        # if self.penalize_sharp_edge > 0 or self.penalize_hard_surface > 0:
        #     loss_dict['sparsity_reg'] = loss_dict['sparsity_reg'] + self._sparsity_regularization(device)

        # RGB loss
        loss_dict['fine_rgb_loss'] = loss_dict['fine_rgb_loss'] + rgb_loss.rgb_loss(
            fine_rgb_map=fine_rgb_map,
            batch_color=batch['color'],
            batch_is_hit=batch['is_hit'],
        )

        # LPIPS loss
        if self.opt.penalize_lpips > 0 and batch['patch_counter'] == 1:
            loss_dict['lpips_loss'] = loss_dict['lpips_loss'] + lpips_loss.lpips_loss(
                lpips_loss_fn=self.lpips_loss_fn,
                fine_rgb_map=fine_rgb_map,
                batch_color=batch['color'],
                penalize_lpips=self.opt.penalize_lpips
            )

        # restart if the network is dead
        if human_out[..., 3].max() <= 0.0:
            self.restart_networks()
            loss_dict = {l: torch.tensor(0.0, requires_grad=True, dtype=torch.float32) for l in
                         LOSS_NAMES}

        loss_dict['rgb_loss'] = loss_dict['fine_rgb_loss'] + loss_dict['color_range_reg'] + loss_dict['lpips_loss']
        loss_dict['can_loss'] = loss_dict['smpl_sym_reg'] + loss_dict['smpl_shape_reg']

        # if we are in the delay phase, only optimize the canonical model, without rgb loss
        # if self.iteration >= self.opt.delay_iters:
        loss_dict['loss'] = loss_dict['rgb_loss'] + loss_dict['can_loss'] + loss_dict['mask_loss'] + \
                                  loss_dict['sparsity_reg']
        # else:
        #     loss_dict['total_loss'] = loss_dict['can_loss'] + loss_dict['mask_loss'] + loss_dict['sparsity_reg']

        return loss_dict
        # if return_rgb:
        #     return loss_dict, fine_rgb_map
        # else:
        #     return loss_dict

    def _eval_human_samples(self, batch):
        """
        Get output of human Nerf + offset nets, by taking a batch of rays,
         converting them to samples, warping them to canonical space,
         and feeding them to Human Nerf

        all parameters are on cuda, including input batch
        """

        human_batch = {
            'origin': batch['origin'],
            'direction': batch['direction'],
            'near': batch['human_near'],
            'far': batch['human_far'],
        }

        # get samples for human rays
        # TODO: ray_to_samples has .to(device) in it
        human_samples = ray_utils.ray_to_samples(
            human_batch,
            self.opt.samples_per_ray,
            perturb=self.opt.perturb
        )

        # human pts, human dirs, human z vals are on cuda
        human_pts = human_samples[0]
        human_dirs = human_samples[1]
        human_z_vals = human_samples[2]
        human_b, human_n, _ = human_pts.shape

        # predict offset, cur_time and offset are on cuda
        cur_time = torch.ones_like(human_pts[..., 0:1]) * batch['cur_view_f']
        # offset = random.choice(self.offset_nets)(torch.cat([human_pts, cur_time], dim=-1))
        offset = self.offset_nets[0](torch.cat([human_pts, cur_time], dim=-1))

        # warp points from observation space to canonical space
        mesh, raw_Ts = self.vertex_forward(int(batch['cap_id']))

        human_pts = human_pts.reshape(-1, 3)

        # new method
        Ts = ray_utils.warp_samples_gpu(pts=human_pts, verts=mesh[0], T=raw_Ts[0])

        # get canonical points and apply offset
        can_pts = (Ts @ ray_utils.to_homogeneous(human_pts)[..., None])[:, :3, 0].reshape(human_b, human_n, 3)
        can_pts = can_pts + offset

        # get canonical directions
        can_dirs = can_pts[:, 1:] - can_pts[:, :-1]
        can_dirs = torch.cat([can_dirs, can_dirs[:, -1:]], dim=1)
        can_dirs = can_dirs / torch.norm(can_dirs, dim=2, keepdim=True)

        # get output of human nerf
        human_out = self.coarse_human_net(can_pts, can_dirs)

        return human_pts, human_dirs, human_z_vals, can_pts, can_dirs, human_out


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

        # get transformation matrices from zero pose to scene pose
        _, T_t2pose = self.hand_model.verts_transformations(global_orient=root_pose, hand_pose=hand_pose, betas=shape,
                                                       transl=trans)
        T_t2pose = T_t2pose.unsqueeze(0)

        return scene_pose_verts, T_t2pose


    def render_rgb_map(self, human_z_vals, human_dirs, human_out, white_bkg):
        # sort the z values in ascending order
        fine_total_zvals, fine_order = torch.sort(human_z_vals, -1)
        fine_total_out = human_out

        # rearrange the rays
        _b, _n, _c = fine_total_out.shape
        fine_total_out = fine_total_out[
            torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
            fine_order.view(_b, _n, 1).repeat(1, 1, _c),
            torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
        ]

        # render the rgb map
        fine_rgb_map, _, _, _, _ = render_utils.raw2outputs(
            fine_total_out,
            fine_total_zvals,
            # was: fine_bkg_dirs[:, 0, :],
            human_dirs[:, 0, :],
            white_bkg=white_bkg
        )
        return fine_rgb_map

    def restart_networks(self):
        print('bad weights, reinitializing')
        self.offset_nets.apply(weight_reset)
        self.coarse_human_net.apply(weight_reset)

