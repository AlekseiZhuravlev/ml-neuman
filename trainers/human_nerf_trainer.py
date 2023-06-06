#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
Canonical human NeRF trainer
'''

import os
import math
import time
import random

import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
import igl
import lpips

from utils import utils, render_utils, ray_utils
from trainers import tensorboard_helper
from options import options
from cameras.captures import ResizedPinholeCapture
from cameras.pinhole_camera import PinholeCamera
from models.vanilla import weight_reset
from utils.constant import HARD_SURFACE_OFFSET, PATCH_SIZE, PATCH_SIZE_SQUARED, CANONICAL_ZOOM_FACTOR, CANONICAL_CAMERA_DIST
import matplotlib.pyplot as plt
import pickle
# import Image class
from PIL import Image

from datasets import ray_generation_from_images
from torch.profiler import profile, record_function, ProfilerActivity



LOSS_NAMES = [
    'fine_rgb_loss',
    'lpips_loss',
    'color_range_reg',
    'smpl_sym_reg',
    'smpl_shape_reg',
    'mask_loss',
    'sparsity_reg'
]


def densepose_name_to_idx():
    return {
        'Torso': [1, 2],
        'Right Hand': [3],
        'Left Hand': [4],
        'Left Foot': [5],
        'Right Foot': [6],
        'Upper Leg Right': [7, 9],
        'Upper Leg Left': [8, 10],
        'Lower Leg Right': [11, 13],
        'Lower Leg Left': [12, 14],
        'Upper Arm Left': [15, 17],
        'Upper Arm Right': [16, 18],
        'Lower Arm Left': [19, 21],
        'Lower Arm Right': [20, 22],
        'Head': [23, 24]
    }


def densepose_idx_to_name():
    name2idx = densepose_name_to_idx()
    idx2name = {}
    for k, v in name2idx.items():
        for item in v:
            idx2name[item] = k
    return idx2name


def turn_smpl_gradient_off(dp_mask):
    assert dp_mask is not None
    grad_mask = np.ones([24, 3])
    idx2name = densepose_idx_to_name()
    visible = [idx2name[i] for i in range(1, 25) if i in np.unique(dp_mask)]
    if 'Upper Leg Left' not in visible:
        grad_mask[1] = 0
    if 'Upper Leg Right' not in visible:
        grad_mask[2] = 0
    if 'Lower Leg Left' not in visible:
        grad_mask[4] = 0
    if 'Lower Leg Right' not in visible:
        grad_mask[5] = 0
    if 'Left Foot' not in visible:
        grad_mask[7] = 0
        grad_mask[10] = 0
    if 'Right Foot' not in visible:
        grad_mask[8] = 0
        grad_mask[11] = 0
    if 'Upper Arm Left' not in visible:
        grad_mask[16] = 0
    if 'Upper Arm Right' not in visible:
        grad_mask[17] = 0
    if 'Lower Arm Left' not in visible:
        grad_mask[18] = 0
    if 'Lower Arm Right' not in visible:
        grad_mask[19] = 0
    if 'Left Hand' not in visible:
        grad_mask[20] = 0
        grad_mask[22] = 0
    if 'Right Hand' not in visible:
        grad_mask[21] = 0
        grad_mask[23] = 0
    if 'Head' not in visible:
        grad_mask[12] = 0
        grad_mask[15] = 0
    return grad_mask.reshape(-1)


class HumanNeRFTrainer:
    def __init__(
        self,
        opt,
        net,
        optimizer,
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        interval_comp=1.0,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.opt = opt
        self.net = net
        self.use_cuda = opt.use_cuda
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.out = opt.out
        if not os.path.exists(opt.out):
            os.makedirs(opt.out)

        self.epoch = 0
        self.iteration = 0
        self.max_iter = opt.max_iter
        self.valid_iter = opt.valid_iter
        self.tb_pusher = tensorboard_helper.TensorboardPusher(opt)
        self.push_opt_to_tb()
        self.need_resume = opt.resume
        if self.need_resume:
            self.resume()
        if self.opt.load_weights:
            self.load_pretrained_weights()

        self.penalize_smpl_alpha = opt.penalize_smpl_alpha
        self.penalize_symmetric_alpha = opt.penalize_symmetric_alpha
        self.penalize_dummy = opt.penalize_dummy
        self.penalize_hard_surface = opt.penalize_hard_surface
        self.penalize_color_range = opt.penalize_color_range
        self.penalize_outside = opt.penalize_outside
        self.penalize_mask = opt.penalize_mask
        self.penalize_lpips = opt.penalize_lpips
        self.penalize_sharp_edge = opt.penalize_sharp_edge

        print('LPIPS is on', next(self.net.parameters()).device)
        if self.penalize_lpips > 0:
            self.lpips_loss_fn = lpips.LPIPS(net='alex').to(next(self.net.parameters()).device)
        self.interval_comp = interval_comp

        # center = mean of wrist, middle1, middle2, middle3
        # up - calculated from code
        center, up = utils.smpl_verts_to_center_and_up(self.val_dataset.scene.static_vert[0])

        render_poses = render_utils.default_360_path(center, up, CANONICAL_CAMERA_DIST, 100)


        if opt.tgt_size is not None:
            render_size = opt.tgt_size
        else:
            render_size = self.val_dataset.scene.captures[0].pinhole_cam.shape
        self.can_caps = [ResizedPinholeCapture(
            PinholeCamera(
                self.val_dataset.scene.captures[0].pinhole_cam.width,
                self.val_dataset.scene.captures[0].pinhole_cam.height,
                CANONICAL_ZOOM_FACTOR * self.val_dataset.scene.captures[0].pinhole_cam.width,
                CANONICAL_ZOOM_FACTOR * self.val_dataset.scene.captures[0].pinhole_cam.width,
                self.val_dataset.scene.captures[0].pinhole_cam.width / 2.0,
                self.val_dataset.scene.captures[0].pinhole_cam.height / 2.0,
            ),
            rp,
            tgt_size=render_size
        ) for rp in render_poses]

        self.ray_generator = ray_generation_from_images.RaysFromImagesGenerator(opt)



    def train(self):
        '''entrance of the whole training process
        '''
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch,
                                 max_epoch,
                                 desc='Train',
                                 ncols=80):
            self.epoch = epoch
            # time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.train_epoch()


            if self.iteration > 100:
                return

            if self.iteration >= self.max_iter:
                break



    def train_epoch(self):
        '''train for one epoch
        one epoch is iterating the whole training dataset once
        '''
        self.net.train()

        loss = 0
        tbar = tqdm.tqdm(enumerate(self.train_loader),
                                              initial=self.iteration % len(
                                                  self.train_loader),
                                              total=len(self.train_loader),
                                              desc=f'Train epoch={self.epoch}, loss={loss:.4f}',
                                              ncols=80,
                                              leave=True,
                                              )
        for batch_idx, data_pack in tbar:

            # Make a plot of the rays

            # for i in range(data_pack['is_bkg'].shape[1]):
            #     print('is_bkg / color', data_pack['is_bkg'][0][i], data_pack['color'][0][i])
            # print('batch_idx', batch_idx)
            # print('data_pack', data_pack)
            # origins = data_pack['direction']
            # colors = data_pack['color']
            # make a 3d plot of the point cloud
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(origins[0, :, 0], origins[0, :, 1], origins[0,:, 2], c=colors[0])
            #
            # with open('/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/input_rays_batch.fig.pickle', 'wb') as f:
            #     pickle.dump(fig, f)

            # plt.savefig('/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/test_rays.png')
            # exit()

            if self.iteration > 10 and self.iteration % self.valid_iter == 0:
                #time.sleep(2)  # Prevent possible deadlock during epoch transition
                with torch.no_grad():
                    self.validate()

            # train for a batch, get total loss
            loss = self.train_batch(data_pack)

            # update progress bar
            tbar.set_description(f'Train epoch={self.epoch}, loss={loss:.4f}')

            if self.iteration > 150:
                return

            if self.iteration >= self.max_iter:
                break
            self.iteration += 1


    def train_batch(self, batch):
        '''train for one batch of data
        '''

        # import torch
        # import gc
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass
        #
        # exit()

        # with profile(activities=[
        #     ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=False, profile_memory=True,
        #         with_modules=True) as prof:
        #
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #     # squeeze batch dim and move to device
        #     for k in raw_batch.keys():
        #         if isinstance(raw_batch[k], torch.Tensor):
        #             raw_batch[k] = raw_batch[k][0].to(device)
        #         # else:
        #         #     (f'raw_batch[{k}] is not a tensor')
        #         elif isinstance(raw_batch[k], np.ndarray):
        #             raise ValueError(f'raw_batch[{k}] is a numpy array')
        #
        #     batch = self.ray_generator.generate_rays_from_images(raw_batch)
        #
        # profiling_data = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
        # print(profiling_data)
        # exit()
        self.optim.zero_grad()

        # calculate losses

        # with profile(activities=[
        #     ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=False, profile_memory=True,
        #         with_modules=True) as prof:

        loss_dict, fine_rgb_map = self.loss_func(batch, return_rgb=True)

        # return 1

        # profiling_data = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
        # print(profiling_data)
        # exit()
        # print('loss_dict', loss_dict)
        # print('fine_rgb_map', fine_rgb_map.shape, fine_rgb_map)

        # origins = batch['direction'].cpu().detach().numpy()
        # colors = fine_rgb_map.cpu().detach().numpy()
        # # make a 3d plot of the point cloud
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c=colors)
        #
        # with open('/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/output_rays_batch.fig.pickle', 'wb') as f:
        #     pickle.dump(fig, f)
        #
        # colors = batch['color'].cpu().detach().numpy()
        # # make a 3d plot of the point cloud
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c=colors)
        #
        # with open('/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/input_rays_batch.fig.pickle', 'wb') as f:
        #     pickle.dump(fig, f)
        #
        # exit()

        # calculate rgb and canonical loss
        loss_dict['rgb_loss'] = loss_dict['fine_rgb_loss'] + loss_dict['color_range_reg'] + loss_dict['lpips_loss']
        loss_dict['can_loss'] = loss_dict['smpl_sym_reg'] + loss_dict['smpl_shape_reg']

        # if we are in the delay phase, only optimize the canonical model, without rgb loss
        if self.iteration >= self.opt.delay_iters:
            loss_dict['total_loss'] = loss_dict['rgb_loss'] + loss_dict['can_loss'] + loss_dict['mask_loss'] + loss_dict['sparsity_reg']
        else:
            loss_dict['total_loss'] = loss_dict['can_loss'] + loss_dict['mask_loss'] + loss_dict['sparsity_reg']


        # if loss is nan, skip this iteration
        if np.isnan(loss_dict['total_loss'].data.item()):
            print('loss is nan during training', loss_dict)
            self.optim.zero_grad()
        else:
            # backprop
            # return 1

            loss_dict['total_loss'].backward()



            # optionally block gradients w.r.t unseen joints
            if self.opt.block_grad:
                try:
                    cap_id = int(batch['cap_id'].item())
                    grad_mask = turn_smpl_gradient_off(
                        self.train_dataset.scene.captures[cap_id].densepose
                    )
                    grad_mask = torch.from_numpy(grad_mask).float().to(
                        next(self.net.parameters()).device
                    )
                    self.net.poses.grad[cap_id] *= grad_mask
                except Exception as e:
                    print('failed to block gradients w.r.t unseen joints')
                    print(e)
                    pass

            losses_no_grad = {k: float(loss_dict[k]) for k in loss_dict.keys()}

            # push training data to tensorboard
            self.push_training_data(
                losses_no_grad,
                self.optim.param_groups[0]['lr']
            )
        self.optim.step()

        # update learning rate
        if self.opt.lrate_decay is not None:
            decay_rate = 0.1
            decay_steps = self.opt.lrate_decay * 1000
            new_lrate = self.opt.learning_rate * (decay_rate ** (self.iteration / decay_steps))
            new_smpl_lrate = self.opt.smpl_lr * (decay_rate ** (self.iteration / decay_steps))
            for param_group in self.optim.param_groups[:1]:
                param_group['lr'] = new_smpl_lrate
            for param_group in self.optim.param_groups[1:3]:
                param_group['lr'] = new_lrate
            ###### update penalty ######
            # reduce prior knowledge based loss
            self.penalize_mask = self.opt.penalize_mask * max(0, 1 - (self.iteration / 60000))
            if self.opt.prior_knowledge_decay:
                self.penalize_symmetric_alpha = self.opt.penalize_symmetric_alpha * max(0, 1 - (self.iteration / 60000))
                self.penalize_dummy = self.opt.penalize_dummy * max(0, 1 - (self.iteration / 60000))
                self.penalize_smpl_alpha = self.opt.penalize_smpl_alpha * max(0, 1 - (self.iteration / 60000))

            assert self.opt.offset_lim >= self.opt.offset_scale >= 0
            new_offset_scale = ((self.opt.offset_lim - self.opt.offset_scale) * max(0, (self.iteration - self.opt.offset_delay) / 60000)) + self.opt.offset_scale
            for _offset_net in self.net.module.offset_nets:
                if self.iteration >= self.opt.offset_delay:
                    _offset_net.nerf.scale = min(new_offset_scale, self.opt.offset_lim)
                else:
                    _offset_net.nerf.scale = 0

        return float(loss_dict['total_loss'])


    def loss_func(self, batch, return_rgb=False):
        """
        Evaluates human nerf on barch, calculates losses
        """

        device = next(self.net.module.parameters()).device

        # initialize loss dictionary
        loss_dict = {l: torch.tensor(0.0, requires_grad=True, device=device, dtype=torch.float32) for l in LOSS_NAMES}

        # remove first axis of a batch
        batch = utils.remove_first_axis(batch)

        # place data on device
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)


        # get results of human nerf
        _, human_dirs, human_z_vals, can_pts, can_dirs, human_out = self._eval_human_samples(batch, device)

        ###########################################################
        # Calculate losses
        ###########################################################

        # canonical space should be symmetric in terms of occupancy
        # if self.penalize_symmetric_alpha > 0:
        #     loss_dict['smpl_sym_reg'] = loss_dict['smpl_sym_reg'] + self._smpl_symmetry_regularization(can_pts, can_dirs, human_out)

        # color of the same point should not change too much due to viewing directions
        # all on cuda
        if self.penalize_color_range > 0:
            # loss_dict['color_range_reg'] = loss_dict['color_range_reg'] + self._color_range_regularization(can_pts, can_dirs, human_out)
            self._color_range_regularization(can_pts, can_dirs, human_out)
        # return 1, 2

        # the rendered human should be close to the detected human mask
        # loosely enforced, the penalty linearly decrease during training
        #
        if self.penalize_mask > 0:
            _, _, human_mask, _, _ = render_utils.raw2outputs(human_out, human_z_vals, human_dirs[:, 0, :], white_bkg=self.opt.white_bkg)
            loss_dict['mask_loss'] = loss_dict['mask_loss'] + F.mse_loss(torch.clamp(human_mask, min=0.0, max=1.0), (1.0-batch['is_bkg'].float())) * self.penalize_mask

        # alpha inside smpl mesh should be 1
        # alpha outside smpl mesh should be 0
        # if self.penalize_smpl_alpha > 0:
        #     loss_dict['smpl_shape_reg'] = loss_dict['smpl_shape_reg'] + self._smpl_shape_regularization(batch, can_pts, can_dirs, human_out)

        # sharp edge loss + hard surface loss
        # USES NUMPY
        # if self.penalize_sharp_edge > 0 or self.penalize_hard_surface > 0:
        #     loss_dict['sparsity_reg'] = loss_dict['sparsity_reg'] + self._sparsity_regularization(device)

        # for l in LOSS_NAMES:
        #     print(l, loss_dict[l].device)
        # exit()

        # RGB loss
        # runs on cuda
        fine_total_zvals, fine_order = torch.sort(human_z_vals, -1)
        fine_total_out = human_out

        _b, _n, _c = fine_total_out.shape
        fine_total_out = fine_total_out[
            torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
            fine_order.view(_b, _n, 1).repeat(1, 1, _c),
            torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
        ]
        fine_rgb_map, _, _, _, _ = render_utils.raw2outputs(
            fine_total_out,
            fine_total_zvals,
            # was: fine_bkg_dirs[:, 0, :],
            human_dirs[:, 0, :],
            white_bkg=self.opt.white_bkg
        )

        hit_index = torch.nonzero(batch['is_hit'])[:, 0]

        loss_dict['fine_rgb_loss'] = loss_dict['fine_rgb_loss'] + F.mse_loss(fine_rgb_map[hit_index], batch['color'][hit_index])

        # LPIPS loss, runs on cuda
        if self.penalize_lpips > 0 and batch['patch_counter'] == 1:
            temp_lpips_loss = self.lpips_loss_fn(
                fine_rgb_map[:PATCH_SIZE_SQUARED].reshape(PATCH_SIZE, PATCH_SIZE, -1).permute(2, 0, 1)*2-1,
                batch['color'][:PATCH_SIZE_SQUARED].reshape(PATCH_SIZE, PATCH_SIZE, -1).permute(2, 0, 1)*2-1) * self.penalize_lpips
            assert torch.numel(temp_lpips_loss) == 1
            loss_dict['lpips_loss'] = loss_dict['lpips_loss'] + temp_lpips_loss.flatten()[0]

        # restart if the network is dead
        if human_out[..., 3].max() <= 0.0:
            print('bad weights, reinitializing')
            self.net.module.offset_nets.apply(weight_reset)
            self.net.module.coarse_human_net.apply(weight_reset)
            loss_dict = {l: torch.tensor(0.0, requires_grad=True, device=device, dtype=torch.float32) for l in LOSS_NAMES}
            # loss_dict = create_loss_dict(device)
        if return_rgb:
            return loss_dict, fine_rgb_map
        else:
            return loss_dict

    def _eval_human_samples(self, batch, device):
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
        human_samples = ray_utils.ray_to_samples(
            human_batch,
            self.opt.samples_per_ray,
            device=device,
            perturb=self.opt.perturb
        )

        # human pts, human dirs, human z vals are on cuda
        human_pts = human_samples[0]
        human_dirs = human_samples[1]
        human_z_vals = human_samples[2]
        human_b, human_n, _ = human_pts.shape

        # predict offset, cur_time and offset are on cuda
        cur_time = torch.ones_like(human_pts[..., 0:1]) * batch['cur_view_f']
        offset = random.choice(self.net.module.offset_nets)(torch.cat([human_pts, cur_time], dim=-1))

        # warp points from observation space to canonical space
        mesh, raw_Ts = self.net.module.vertex_forward(int(batch['cap_id']))

        human_pts = human_pts.reshape(-1, 3)

        # new method
        Ts = ray_utils.warp_samples_gpu(pts=human_pts, verts=mesh[0], T=raw_Ts[0])

        # get canonical points and apply offset
        can_pts = (Ts @ ray_utils.to_homogeneous(human_pts)[..., None])[:, :3, 0].reshape(human_b, human_n, 3)
        can_pts += offset

        # get canonical directions
        can_dirs = can_pts[:, 1:] - can_pts[:, :-1]
        can_dirs = torch.cat([can_dirs, can_dirs[:, -1:]], dim=1)
        can_dirs = can_dirs / torch.norm(can_dirs, dim=2, keepdim=True)

        # get output of human nerf
        human_out = self.net.module.coarse_human_net(can_pts, can_dirs)

        return human_pts, human_dirs, human_z_vals, can_pts, can_dirs, human_out

    ###########################################################
    # Losses
    ###########################################################
    def _color_range_regularization(self, pts, dirs, tgts):
        device = pts.device
        dummy_dirs = torch.randn(dirs.shape, dtype=dirs.dtype, device=device)
        dummy_dirs = dummy_dirs / torch.norm(dummy_dirs, dim=-1, keepdim=True)
        dummy_out = self.net.module.coarse_human_net(pts, dummy_dirs)
        color_reg = F.mse_loss(
            torch.sigmoid(dummy_out.reshape(-1, 4))[:, :3],
            torch.sigmoid(tgts.reshape(-1, 4))[:, :3]
        ) * self.penalize_color_range
        return color_reg

    def _smpl_symmetry_regularization(self, pts, dirs, tgts):
        '''
        we use dummy ray directions for the flipped points, since we only
        care about the occupancy symmetry.
        '''
        pts_flip = pts.clone().detach()
        pts_flip[..., 0] *= -1
        out_flip = self.net.module.coarse_human_net(pts_flip, dirs.clone().detach())
        sym_reg = F.mse_loss(
            torch.tanh(torch.relu(tgts[..., 3])),
            torch.tanh(torch.relu(out_flip[..., 3]))
        ) * self.penalize_symmetric_alpha
        return sym_reg

    def _smpl_shape_regularization(self, batch, pts, dirs, pred):
        device = pts.device
        smpl_reg = torch.tensor(0.0, requires_grad=True).float().to(device)
        can_mesh = self.val_dataset.scene.captures[batch['cap_id']].can_mesh


        # TODO change to torch
        dist_human, _, _ = igl.signed_distance(
            pts.reshape(-1, 3).detach().cpu().numpy(),
            can_mesh.verts_packed().cpu().numpy(),
            can_mesh.faces_packed().cpu().numpy(),
        )
        print('can_mesh', type(can_mesh))#can_mesh.shape, can_mesh)
        print('pts.reshape(-1, 3)', pts.reshape(-1, 3).shape, pts.reshape(-1, 3))
        print('can_mesh.verts_packed()', can_mesh.verts_packed().shape, can_mesh.verts_packed())
        print('can_mesh.faces_packed()', can_mesh.faces_packed().shape, can_mesh.faces_packed())
        print('dist_human', dist_human.shape)
        print(dist_human)

        raise NotImplementedError('TODO change to torch')
        exit()



        inside_volume = dist_human < 0
        if inside_volume.sum() > 0:
            smpl_reg = smpl_reg + F.mse_loss(
                1 - torch.exp(-torch.relu(pred.reshape(-1, 4)[inside_volume][:, 3])),
                torch.ones_like(pred.reshape(-1, 4)[inside_volume][:, 3])
            ) * self.penalize_smpl_alpha

        # generate random samples inside a box in canonical space
        if self.penalize_dummy > 0:
            dummy_pts = (torch.rand(pts.shape, dtype=pts.dtype, device=device) - 0.5) * 3
            dummy_out = self.net.module.coarse_human_net(dummy_pts, dirs)

            # TODO change to torch
            dist_dummy, _, _ = igl.signed_distance(
                dummy_pts.reshape(-1, 3).detach().cpu().numpy(),
                can_mesh.verts_packed().cpu().numpy(),
                can_mesh.faces_packed().cpu().numpy(),
            )
            dummy_inside = dist_dummy < 0
            dummy_outside = dist_dummy > 0
            if dummy_inside.sum() > 0:
                smpl_reg = smpl_reg + F.mse_loss(
                    1 - torch.exp(-torch.relu(dummy_out.reshape(-1, 4)[dummy_inside][:, 3])),
                    torch.ones_like(dummy_out.reshape(-1, 4)[dummy_inside][:, 3])
                ) * self.penalize_dummy
            if dummy_outside.sum() > 0:
                smpl_reg = smpl_reg + F.l1_loss(
                    (1 - torch.exp(-torch.relu(dummy_out.reshape(-1, 4)[dummy_outside][:, 3]))) * torch.pow(torch.abs(torch.from_numpy(dist_dummy[dummy_outside]).to(device)) * self.opt.penalize_outside_factor, self.opt.dist_exponent),
                    torch.zeros_like(dummy_out.reshape(-1, 4)[dummy_outside][:, 3])
                ) * self.penalize_dummy
        return smpl_reg

    def _sparsity_regularization(self, device):
        sparsity_reg = torch.tensor(0.0, requires_grad=True, device=device)
        # pick a random camera
        num_can_rays = 128
        can_cap = random.choice(self.can_caps)
        coords = np.argwhere(np.ones(can_cap.shape))
        coords = coords[np.random.randint(0, len(coords), num_can_rays)][:, ::-1]  # could get duplicated rays
        can_orig, can_dir = ray_utils.shot_rays(can_cap, coords)
        can_pts, can_dirs, can_z_vals = ray_utils.ray_to_samples(
            {
                'origin':    torch.from_numpy(can_orig).float().to(device),
                'direction': torch.from_numpy(can_dir).float().to(device),
                'near':      torch.zeros(num_can_rays, 1).float().to(device),
                'far':       torch.ones(num_can_rays, 1).float().to(device) * CANONICAL_CAMERA_DIST * 1.667,
            },
            samples_per_ray=self.opt.samples_per_ray,
            device=device,
            perturb=self.opt.perturb
        )
        can_out = self.net.module.coarse_human_net(can_pts, can_dirs)
        # compensate the interval difference between observation space and canonical space
        can_out[..., -1] *= self.interval_comp
        _, _, can_mask, can_weights, _ = render_utils.raw2outputs(can_out, can_z_vals.clone(), can_dirs[:, 0, :].clone(), white_bkg=True)
        can_weights = torch.clip(can_weights, 0.0, 1.0)
        can_mask = torch.clip(can_mask, 0.0, 1.0)
        # sharp edge loss
        if self.penalize_sharp_edge > 0:
            sparsity_reg = sparsity_reg + torch.mean(-torch.log(
                torch.exp(-torch.abs(can_mask)) + torch.exp(-torch.abs(1-can_mask))
            ) + HARD_SURFACE_OFFSET) * self.penalize_sharp_edge
        # hard surface loss
        if self.penalize_hard_surface > 0:
            sparsity_reg = sparsity_reg + torch.mean(-torch.log(
                torch.exp(-torch.abs(can_weights)) + torch.exp(-torch.abs(1-can_weights))
            ) + HARD_SURFACE_OFFSET) * self.penalize_hard_surface
        return sparsity_reg


    ########################################################################################################################
    # validation and logging
    ########################################################################################################################

    def validate_batch(self, raw_batch):
        self.optim.zero_grad()
        assert self.net.training is False

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # squeeze batch dim and move to device
        for k in raw_batch.keys():
            if isinstance(raw_batch[k], torch.Tensor):
                raw_batch[k] = raw_batch[k][0].to(device)
            else:
                raise ValueError(f'raw_batch[{k}] is not a tensor')

        batch = self.ray_generator.generate_rays_from_images(raw_batch)
        print('validation batch', batch)
        exit()

        # print('validate_batch', batch['color'].shape)
        with torch.no_grad():
            loss_dict, rgb_map = self.loss_func(batch, return_rgb=True)

            # print('rgb_map', rgb_map.shape, rgb_map)

            # get current timestamp
            ts = time.time()

            origins = batch['direction'].cpu().detach().numpy()
            colors = rgb_map.cpu().detach().numpy()
            # clip colors to [0, 1]
            colors = np.clip(colors, 0.0, 1.0)

            # make a 3d plot of the point cloud
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c=colors)

            with open(f'/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/output_rays_batch_{int(ts / 1000)}.fig.pickle',
                      'wb') as f:
                pickle.dump(fig, f)

            colors = batch['color'].cpu().detach().numpy()
            colors = np.clip(colors, 0.0, 1.0)
            # make a 3d plot of the point cloud
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c=colors)

            with open(f'/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/input_rays_batch_{int(ts / 1000)}.fig.pickle',
                      'wb') as f:
                pickle.dump(fig, f)

            # exit()

            loss_dict['rgb_loss'] = loss_dict['fine_rgb_loss'] + loss_dict['color_range_reg'] + loss_dict['lpips_loss']
            loss_dict['can_loss'] = loss_dict['smpl_sym_reg'] + loss_dict['smpl_shape_reg']
            loss_dict['total_loss'] = loss_dict['rgb_loss'] + loss_dict['can_loss'] + loss_dict['mask_loss'] + loss_dict['sparsity_reg']
        return {k: v.data.item() for k, v in loss_dict.items()}

    def validate(self):
        '''validate for whole validation dataset
        '''
        training = self.net.training
        self.net.eval()
        all_loss = {l: [] for l in LOSS_NAMES}
        all_loss['rgb_loss'] = []
        all_loss['can_loss'] = []
        all_loss['total_loss'] = []
        for batch_idx, batch in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            cur_loss = self.validate_batch(batch)
            for k, v in cur_loss.items():
                all_loss[k].append(v)

        # draw visualizations
        overfit_id = self.train_dataset.scene.fname_to_index_dict[self.train_dataset.inclusions[len(self.train_dataset.inclusions) // 2]]
        overfit_cap = self.train_dataset.scene.captures[overfit_id]
        verts, _ = self.net.vertex_forward(overfit_id)
        verts = verts[0]
        faces = torch.from_numpy(self.val_dataset.scene.faces[:, :3]).to(verts.device)
        overlay = render_utils.overlay_smpl(overfit_cap.image, verts, faces, overfit_cap)

        if self.opt.random_view:
            can_cap = random.choice(self.can_caps)
        else:
            can_cap = self.can_caps[0]
        rgb_map, depth_map, acc_map = render_utils.render_smpl_nerf(
            self.net,
            can_cap,
            self.val_dataset.scene.static_vert[0],
            self.val_dataset.scene.faces,
            None,
            rays_per_batch=self.opt.rays_per_batch,
            samples_per_ray=self.opt.samples_per_ray,
            white_bkg=True,
            render_can=True,
            return_mask=True,
            return_depth=True,
            interval_comp=self.interval_comp
        )
        try:
            alpha_mask = acc_map >= 0.9999999999
            d_min = depth_map[alpha_mask].min()
            d_max = depth_map[alpha_mask].max()
            depth_map[depth_map <= d_min] = d_min
            depth_map[depth_map >= d_max] = d_max
        except:
            pass
        acc_map = np.stack([acc_map]*3, -1)
        depth_map = np.stack([depth_map]*3, -1)

        validation_data = {k: np.array(v).mean() for k, v in all_loss.items()}
        validation_data['render'] = utils.np_img_to_torch_img(np.stack([rgb_map, depth_map, acc_map, overlay]))
        self.push_validation_data(validation_data)
        self.save_model()
        if training:
            self.net.train()

    def save_model(self):
        save_dict = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'optim_state_dict': self.optim.state_dict(),
            'hybrid_model_state_dict': self.net.state_dict(),
        }
        torch.save(save_dict, os.path.join(self.out, 'checkpoint.pth.tar'))

    def push_validation_data(self, validation_data):
        render = vutils.make_grid(validation_data['render'], nrow=2, normalize=True, scale_each=True)
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(False)
        tb_datapack.set_iteration(self.iteration)
        for key in validation_data.keys():
            if 'loss' in key or 'reg' in key:
                if key == 'lpips_loss' and validation_data[key] == 0:
                    pass
                else:
                    tb_datapack.add_scalar({f'val_loss/{key}': validation_data[key]})
        tb_datapack.add_image({'render/val': render})
        self.tb_pusher.push_to_tensorboard(tb_datapack)


    def push_training_data(self, losses, lr):
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(True)
        tb_datapack.set_iteration(self.iteration)
        for key in losses.keys():
            if 'loss' in key or 'reg' in key:
                if key == 'lpips_loss' and losses[key] == 0:
                    pass
                else:
                    tb_datapack.add_scalar({f'train_loss/{key}': losses[key]})
        tb_datapack.add_scalar({'lr/lr': lr})
        tb_datapack.add_scalar({'hyper_params/offset_scale': self.net.module.offset_nets[0].nerf.scale})
        tb_datapack.add_scalar({'hyper_params/penalize_mask': self.penalize_mask})
        tb_datapack.add_scalar({'hyper_params/penalize_symmetric_alpha': self.penalize_symmetric_alpha})
        tb_datapack.add_scalar({'hyper_params/penalize_dummy': self.penalize_dummy})
        tb_datapack.add_scalar({'hyper_params/penalize_smpl_alpha': self.penalize_smpl_alpha})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    def resume(self):
        '''resume training:
        resume from the recorded epoch, iteration, and saved weights.
        resume from the model with the same name.
        '''
        if hasattr(self.opt, 'load_weights'):
            assert self.opt.load_weights is None or self.opt.load_weights == False
        # 1. load check point
        checkpoint_path = os.path.join(self.opt.out, 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        else:
            raise FileNotFoundError(f'model check point cannot found: {checkpoint_path}')
        # 2. load data
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        self.load_pretrained_weights()
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
        utils.print_notification([f'Resuming from {self.iteration} iterations'])

    def load_pretrained_weights(self):
        '''
        load pretrained weights from another model
        '''
        # if hasattr(self.opt, 'resume'):
        #     assert self.opt.resume is False
        assert os.path.isfile(self.opt.load_weights_path), self.opt.load_weights_path
        content_list = []
        saved = torch.load(self.opt.load_weights_path, map_location='cpu')
        utils.safe_load_weights(self.net, saved['hybrid_model_state_dict'])
        content_list += [f'Loaded pretrained weights from {self.opt.load_weights_path}']
        utils.print_notification(content_list)

    def push_opt_to_tb(self):
        opt_str = options.opt_to_string(self.opt)
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(False)
        tb_datapack.set_iteration(self.iteration)
        tb_datapack.add_text({'options': opt_str})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

