import igl
import lpips
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import os
import pickle
import random
import time
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.utils as vutils
import tqdm
# import Image class
from PIL import Image
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data.distributed import DistributedSampler

from cameras.captures import ResizedPinholeCapture
from cameras.pinhole_camera import PinholeCamera
from models.vanilla import weight_reset
from options import options
from trainers import tensorboard_helper
from utils import utils, render_utils, ray_utils
from utils.constant import HARD_SURFACE_OFFSET, PATCH_SIZE, PATCH_SIZE_SQUARED, CANONICAL_ZOOM_FACTOR, \
    CANONICAL_CAMERA_DIST

import lightning_code.pytorch as pl


LOSS_NAMES = [
    'fine_rgb_loss',
    'lpips_loss',
    'color_range_reg',
    'smpl_sym_reg',
    'smpl_shape_reg',
    'mask_loss',
    'sparsity_reg'
]


class HumanNeRFTrainer:
    def __init__(
            self,
            opt,
            model,
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
        # self.model = model
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

        #################################################################
        # DDP related parameters
        #################################################################

        self.gpu_id = 0
        # self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        # self.model = DDP(self.model, device_ids=[self.gpu_id])

        print(f"GPU ID: {self.gpu_id}")
        # exit()

    def ddp_setup(self):
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    def train(self):
        """
        entrance of the whole training process
        """
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch,
                                 max_epoch,
                                 desc='Train',
                                 ncols=80):
            self.epoch = epoch
            self.train_epoch()

            if self.iteration > 150:
                return

            if self.iteration >= self.max_iter:
                break

    def train_epoch(self):
        """
        train for one epoch
        one epoch is iterating the whole training dataset once
        """
        self.model.train()

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

            if self.iteration > 10 and self.iteration % self.valid_iter == 0:
                with torch.no_grad():
                    self.validate()

            # train for a batch, get total loss
            loss = self.train_batch(data_pack)

            # update progress bar
            tbar.set_description(f'Train epoch={self.epoch}, loss={loss:.4f}')

            self.iteration += 1

    def train_batch(self, batch):
        """
        train for one batch of data
        """

        # remove first axis of a batch
        batch = utils.remove_first_axis(batch)

        # place data on device
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.gpu_id)

        # zero gradients
        self.optim.zero_grad()

        # calculate losses
        loss_dict, fine_rgb_map = self.loss_func(batch, return_rgb=True)

        # calculate rgb and canonical loss
        loss_dict['rgb_loss'] = loss_dict['fine_rgb_loss'] + loss_dict['color_range_reg'] + loss_dict['lpips_loss']
        loss_dict['can_loss'] = loss_dict['smpl_sym_reg'] + loss_dict['smpl_shape_reg']

        # if we are in the delay phase, only optimize the canonical model, without rgb loss
        if self.iteration >= self.opt.delay_iters:
            loss_dict['total_loss'] = loss_dict['rgb_loss'] + loss_dict['can_loss'] + loss_dict['mask_loss'] + \
                                      loss_dict['sparsity_reg']
        else:
            loss_dict['total_loss'] = loss_dict['can_loss'] + loss_dict['mask_loss'] + loss_dict['sparsity_reg']

        # if loss is nan, skip this iteration
        if not torch.isnan(loss_dict['total_loss'].data.item()):
            # backprop
            loss_dict['total_loss'].backward()

            # optionally block gradients w.r.t unseen joints
            if self.opt.block_grad:
                try:
                    cap_id = int(batch['cap_id'].item())
                    grad_mask = turn_smpl_gradient_off(
                        self.train_dataset.scene.captures[cap_id].densepose
                    )
                    grad_mask = torch.from_numpy(grad_mask).float().to(
                        #next(self.model.parameters()).device
                        self.gpu_id
                    )
                    self.model.poses.grad[cap_id] *= grad_mask
                except Exception as e:
                    print('failed to block gradients w.r.t unseen joints')
                    print(e)
                    pass

            #######################################################################
            # push losses to tensorboard
            #######################################################################
            losses_no_grad = {k: float(loss_dict[k]) for k in loss_dict.keys()}

            # push training data to tensorboard
            self.push_training_data(
                losses_no_grad,
                self.optim.param_groups[0]['lr']
            )
        else:
            print('loss is nan during training', loss_dict)
            self.optim.zero_grad()

        # update parameters
        self.optim.step()

        # update learning rate
        self.update_learning_rate()

        return float(loss_dict['total_loss'])

    def update_learning_rate(self):
        if self.opt.lrate_decay:
            # learining rate decay rate and steps
            decay_rate = 0.1
            decay_steps = self.opt.lrate_decay * 1000

            # update learning rate for networks
            new_lrate = self.opt.learning_rate * (decay_rate ** (self.iteration / decay_steps))
            for param_group in self.optim.param_groups[1:3]:
                param_group['lr'] = new_lrate

            # update smpl learning rate
            new_smpl_lrate = self.opt.smpl_lr * (decay_rate ** (self.iteration / decay_steps))
            for param_group in self.optim.param_groups[:1]:
                param_group['lr'] = new_smpl_lrate

            ###### update penalty ######

            # update penalty for mask
            self.model.opt.penalize_mask = self.opt.penalize_mask * max(0, 1 - (self.iteration / 60000))

            # reduce prior knowledge based loss
            if self.opt.prior_knowledge_decay:
                self.model.opt.penalize_symmetric_alpha = self.opt.penalize_symmetric_alpha * max(0, 1 - (
                            self.iteration / 60000))
                self.model.opt.penalize_dummy = self.opt.penalize_dummy * max(0, 1 - (self.iteration / 60000))
                self.model.opt.penalize_smpl_alpha = self.opt.penalize_smpl_alpha * max(0, 1 - (self.iteration / 60000))

            assert self.opt.offset_lim >= self.opt.offset_scale >= 0

            # update offset scale
            new_offset_scale = (
                                       (self.opt.offset_lim - self.opt.offset_scale) *
                                       max(0, (self.iteration - self.opt.offset_delay) / 60000)
                               ) + self.opt.offset_scale

            for i in range(len(self.model.offset_nets)):
                # for each offset net, update the scale
                if self.iteration >= self.opt.offset_delay:
                    self.model.offset_nets[i].nerf.scale = min(new_offset_scale, self.opt.offset_lim)
                else:
                    self.model.offset_nets[i].nerf.scale = 0

    ########################################################################################################################
    # validation and logging
    ########################################################################################################################
