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
    # validation
    ########################################################################################################################

    def validate_batch(self, raw_batch):
        self.optim.zero_grad()
        assert self.model.training is False

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

            with open(
                    f'/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/output_rays_batch_{int(ts / 1000)}.fig.pickle',
                    'wb') as f:
                pickle.dump(fig, f)

            colors = batch['color'].cpu().detach().numpy()
            colors = np.clip(colors, 0.0, 1.0)
            # make a 3d plot of the point cloud
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c=colors)

            with open(
                    f'/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/input_rays_batch_{int(ts / 1000)}.fig.pickle',
                    'wb') as f:
                pickle.dump(fig, f)

            # exit()

            loss_dict['rgb_loss'] = loss_dict['fine_rgb_loss'] + loss_dict['color_range_reg'] + loss_dict['lpips_loss']
            loss_dict['can_loss'] = loss_dict['smpl_sym_reg'] + loss_dict['smpl_shape_reg']
            loss_dict['total_loss'] = loss_dict['rgb_loss'] + loss_dict['can_loss'] + loss_dict['mask_loss'] + \
                                      loss_dict['sparsity_reg']
        return {k: v.data.item() for k, v in loss_dict.items()}

    def validate(self):
        '''validate for whole validation dataset
        '''
        training = self.model.training
        self.model.eval()
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
        overfit_id = self.train_dataset.scene.fname_to_index_dict[
            self.train_dataset.inclusions[len(self.train_dataset.inclusions) // 2]]
        overfit_cap = self.train_dataset.scene.captures[overfit_id]
        verts, _ = self.model.vertex_forward(overfit_id)
        verts = verts[0]
        faces = torch.from_numpy(self.val_dataset.scene.faces[:, :3]).to(verts.device)
        overlay = render_utils.overlay_smpl(overfit_cap.image, verts, faces, overfit_cap)

        if self.opt.random_view:
            can_cap = random.choice(self.can_caps)
        else:
            can_cap = self.can_caps[0]
        rgb_map, depth_map, acc_map = render_utils.render_smpl_nerf(
            self.model,
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
        acc_map = np.stack([acc_map] * 3, -1)
        depth_map = np.stack([depth_map] * 3, -1)

        validation_data = {k: np.array(v).mean() for k, v in all_loss.items()}
        validation_data['render'] = utils.np_img_to_torch_img(np.stack([rgb_map, depth_map, acc_map, overlay]))
        self.push_validation_data(validation_data)
        self.save_model()
        if training:
            self.model.train()

    #################################################################
    # Tensorboard
    #################################################################

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
        tb_datapack.add_scalar({'hyper_params/offset_scale': self.model.offset_nets[0].nerf.scale})
        tb_datapack.add_scalar({'hyper_params/penalize_mask': self.model.opt.penalize_mask})
        tb_datapack.add_scalar({'hyper_params/penalize_symmetric_alpha': self.model.opt.penalize_symmetric_alpha})
        tb_datapack.add_scalar({'hyper_params/penalize_dummy': self.model.opt.penalize_dummy})
        tb_datapack.add_scalar({'hyper_params/penalize_smpl_alpha': self.model.opt.penalize_smpl_alpha})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    def push_opt_to_tb(self):
        opt_str = options.opt_to_string(self.opt)
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(False)
        tb_datapack.set_iteration(self.iteration)
        tb_datapack.add_text({'options': opt_str})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    #################################################################
    # Saving checkpoints
    #################################################################

    def save_model(self):
        save_dict = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'optim_state_dict': self.optim.state_dict(),
            'hybrid_model_state_dict': self.model.state_dict(),
        }
        torch.save(save_dict, os.path.join(self.out, 'checkpoint.pth.tar'))

    #################################################################
    # Loading checkpoints
    #################################################################

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
        utils.safe_load_weights(self.model, saved['hybrid_model_state_dict'])
        content_list += [f'Loaded pretrained weights from {self.opt.load_weights_path}']
        utils.print_notification(content_list)
