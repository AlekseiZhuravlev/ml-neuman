#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import argparse
import os
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_io import neuman_helper
from utils import utils
from datasets import background_rays, human_rays, human_images_only, ray_generation_from_images
from options.options import str2bool
from options import options
from models import vanilla, human_nerf
from trainers import vanilla_nerf_trainer, human_nerf_trainer
from torch.profiler import profile, record_function, ProfilerActivity

from lightning_code import model as lightning_model
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from lightning.pytorch.profilers import AdvancedProfiler

def train_human(opt):

    train_scene = neuman_helper.NeuManReader.read_scene(
        opt.scene_dir,
        tgt_size=opt.tgt_size,
        normalize=opt.normalize,
        bkg_range_scale=opt.bkg_range_scale,
        human_range_scale=opt.human_range_scale,
        mask_dir=opt.mask_dir,
        smpl_type=opt.smpl_type
    )
    train_split, val_split, _ = neuman_helper.create_split_files(train_scene, opt.scene_dir)

    opt.geo_threshold = 1
    # if opt.geo_threshold < 0:
    #     can_bones = []
    #     bones = []
    #     for i in range(len(train_scene.captures)):
    #         bones.append(np.linalg.norm(train_scene.smpls[i]['joints_3d'][3] - train_scene.smpls[i]['joints_3d'][0]))
    #         can_bones.append(np.linalg.norm(train_scene.smpls[i]['static_joints_3d'][3] - train_scene.smpls[i]['static_joints_3d'][0]))
    #     opt.geo_threshold = np.mean(bones) / np.mean(can_bones)

    poses = np.stack([item['pose'] for item in train_scene.smpls])
    betas = np.stack([item['betas'] for item in train_scene.smpls])
    transes = np.stack([item['trans'] for item in train_scene.smpls])

    # create main network
    # net = human_nerf.HumanNeRF(opt, poses.copy(), betas.copy(), transes.copy(), scale=train_scene.scale)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_scene.read_data_to_ram(data_list=['image', 'depth'])

    # runs binary mask dilation on the border of the mask
    utils.add_border_mask(train_scene, iterations=opt.dilation)

    utils.add_pytorch3d_cache(train_scene, device)

    # put scene.verts, scene.Ts to device AND cpu
    utils.move_smpls_to_torch(train_scene, device)

    # near_far cache will be created
    train_dset = human_rays.HumanRayDataset(
        opt,
        train_scene,
        'train',
        train_split,
        n_repeats=8
    )
    val_dset = human_rays.HumanRayDataset(
        opt,
        train_scene,
        'val',
        val_split,
        near_far_cache=train_dset.near_far_cache,
        n_repeats=1
    )

    print(f'os.cpu_count {os.cpu_count()}, len(os.sched_getaffinity(0)) {len(os.sched_getaffinity(0))}')

    # worker_count = min(os.cpu_count(), len(os.sched_getaffinity(0))) // torch.cuda.device_count()
    worker_count = torch.cuda.device_count() * 5 if torch.cuda.device_count() else 5
    # print(f'worker_count {worker_count}')
    # worker_count = 0

    train_loader = DataLoader(
        train_dset,
        batch_size=1,
        shuffle=True,
        num_workers=worker_count,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=1,
        num_workers=worker_count,
        # pin_memory=True,
    )


    logger = TensorBoardLogger(opt.out_dir + '/' + opt.name)
    checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor="epoch", mode='max', every_n_epochs=1)
    # profiler = AdvalcedProfiler(
    #     dirpath="/home/azhuavlev/Desktop/Results/neuman-mano/interhand/",
    #     filename="perf_logs.txt"
    # )
    # stats_monitor = DeviceStatsMonitor(cpu_stats=False)

    restart_exception = True
    while restart_exception:
        print('Restarting training')
        restart_exception = False
        try:
            model = lightning_model.HumanNeRF(opt, poses.copy(), betas.copy(), transes.copy(), scale=train_scene.scale)

            trainer = L.Trainer(
                max_epochs=1000,
                benchmark=True,
                logger=logger,
                default_root_dir=opt.out_dir + '/' + opt.name,
                check_val_every_n_epoch=5,
                log_every_n_steps=100,
                # profiler=profiler,
                # profiler='simple',
                callbacks=[
                    checkpoint_callback,
                    # stats_monitor,
                ],
            )
            trainer.fit(
                model,
                train_loader,
                val_loader
            )
        except NotImplementedError as e:
            print(
                """
                !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                Exception
                !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                """
            )
            del model, trainer

            print(e)
            restart_exception = True


    # TODO do we need to optimize trans?
    # assert opt.bkg_lr == 0
    # if opt.train_mode == 'smpl_only':
    #     assert opt.offset_scale == 0
    #     optim_list = [
    #         {"params": net.poses, "lr": opt.learning_rate},
    #         {"params": net.coarse_human_net.parameters(), "lr": opt.learning_rate},
    #     ]
    # elif opt.train_mode == 'smpl_and_offset':
    #     optim_list = [
    #         {"params": net.poses, "lr": opt.smpl_lr},
    #         {"params": net.coarse_human_net.parameters(), "lr": opt.learning_rate},
    #         {"params": net.offset_nets.parameters(), "lr": opt.learning_rate},
    #     ]
    # optim = torch.optim.Adam(optim_list)

    # trainer = human_nerf_trainer.HumanNeRFTrainer(
    #     opt,
    #     net,
    #     optim,
    #     train_loader,
    #     val_loader,
    #     train_dset,
    #     val_dset,
    #     interval_comp=opt.geo_threshold
    # )

    # torch.backends.cudnn.benchmark = True
    # torch.jit.enable_onednn_fusion(True)

    # trainer.ddp_setup()

    # trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options.set_general_option(parser)
    options.set_nerf_option(parser)
    options.set_pe_option(parser)
    parser.add_argument('--train_mode', type=str, default='bkg', choices=['bkg', 'smpl_only', 'smpl_and_offset'])
    opt, _ = parser.parse_known_args()

    # common args with diferent defaults
    parser.add_argument('--rays_per_batch', default=2048, type=int, help='how many rays per batch')
    parser.add_argument('--valid_iter', type=int, default=1000, help='interval of validation')
    parser.add_argument('--max_iter', type=int, default=300000, help='total training iterations')
    parser.add_argument('--body_rays_ratio', default=0.95, type=float, help='the percentage of rays on body')
    parser.add_argument('--border_rays_ratio', default=0.05, type=float, help='the percentage of rays on human border')
    parser.add_argument('--bkg_rays_ratio', default=0.0, type=float, help='the percentage of rays on background')
    parser.add_argument('--perturb', default=1, type=float, help='perturbation on samples location')

    # specific args for human
    parser.add_argument('--bkg_lr', default=0, type=float, help='background model learning rate')
    parser.add_argument('--smpl_lr', default=3e-4, type=float, help='SMPL parameters learning rate')
    parser.add_argument('--geo_threshold', default=-1, type=float, help='geometry threshold for pruning far away volumes from SMPL mesh')
    parser.add_argument('--penalize_smpl_alpha', default=1, type=float, help='penalize samples inside SMPL to be opaque otherwise transparent in canonical space')
    parser.add_argument('--penalize_outside', default=True, type=str2bool, help='only penalize points inside SMPL')
    parser.add_argument('--penalize_outside_factor', default=2.0, type=float, help='penalty factor')
    parser.add_argument('--penalize_outside_loss', default='l1', type=str, help='l1 or mse')
    parser.add_argument('--dist_exponent', default=1.0, type=float, help='distance exponent for weighing the loss')
    parser.add_argument('--penalize_symmetric_alpha', default=0.1, type=float, help='enforcing the canonical space to be symmetric')
    parser.add_argument('--penalize_hard_surface', default=0.1, type=float, help='penalize with hard surface loss')
    parser.add_argument('--penalize_dummy', default=1.0, type=float, help='generate random dummy points in canonical space(see penalize_smpl_alpha)')
    parser.add_argument('--penalize_color_range', default=0.1, type=float, help='generate random dummy points in canonical space(see penalize_smpl_alpha)')
    parser.add_argument('--penalize_mask', default=0.01, type=float, help='human mask')
    parser.add_argument('--penalize_sharp_edge', default=0.1, type=float, help='canonical mask')
    parser.add_argument('--penalize_lpips', default=0.01, type=float, help='train with patch')
    parser.add_argument('--chunk', default=10000, type=int, help='chunk size per caching iteration')
    parser.add_argument('--load_background', type=str, default=None, help='load a pretrained background, for joint training')
    parser.add_argument('--load_can', type=str, default=None, help='load a pretrained canonical volume')

    parser.add_argument('--num_offset_nets', default=1, type=int, help='how many offset networks')
    parser.add_argument('--offset_scale', default=0, type=float, help='scale the predicted offset')
    parser.add_argument('--offset_scale_type', default='linear', type=str, help='no/linear/tanh')
    parser.add_argument('--offset_lim', default=1.0, type=float, help='cap the scale of predicted offset')
    parser.add_argument('--offset_delay', default=20000, type=int, help='delay the offset net training')
    parser.add_argument('--prior_knowledge_decay', type=str2bool, default=False, help='reduce prior knowledge based loss')
    parser.add_argument('--block_grad', type=str2bool, default=True, help='block gradients w.r.t occluded joints')
    parser.add_argument('--random_view', type=str2bool, default=False, help='random view point for visualization')

    # scene-related args
    parser.add_argument('--scene_dir', type=str, default=None, required=True)
    parser.add_argument('--normalize', type=str2bool, default=True, required=False, help='normalize the scene based on scene bounds')
    parser.add_argument('--bkg_range_scale', default=3, type=float, help='extend near/far range for background')
    parser.add_argument('--human_range_scale', default=1.5, type=float, help='extend near/far range for human')
    parser.add_argument('--image_height', type=int, default=None, required=False)
    parser.add_argument('--image_width', type=int, default=None, required=False)
    parser.add_argument('--white_bkg', type=str2bool, default=True, required=False)

    parser.add_argument('--samples_per_ray', default=32, type=int, help='how many samples per ray')
    parser.add_argument('--importance_samples_per_ray', default=32, type=int, help='how many importance samples per ray')

    parser.add_argument('--delay_iters', default=0, type=int, help='delay RGB loss, train with alpha/depth first')
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='NeRF learning rate')
    parser.add_argument('--lrate_decay', default=250, type=int, help='NeRF learning rate decay')
    parser.add_argument('--raw_noise_std', default=0, type=float, help='add noise while rendering')

    # directories
    parser.add_argument('--out_dir', default='/home/azhuavlev/Desktop/Results/neuman-mano/', type=str, help='output dir')
    parser.add_argument('--name', default='dummy', type=str, help='name of run')
    parser.add_argument('--resume', type=str2bool, default=False, help='resume training with same model name')
    parser.add_argument('--load_weights', type=str, default=None, help='load a pretrained set of weights, you need to provide the model id')
    parser.add_argument('--mask_dir', default='segmentations', type=str, help='mask folder')
    parser.add_argument('--smpl_type', default='optimized', type=str, choices=['romp', 'optimized'], help='smpl source')
    parser.add_argument('--dilation', type=int, default=30, help='mask dilation')

    opt = parser.parse_args()
    if opt.image_height is not None or opt.image_width is not None:
        assert opt.image_height is not None
        assert opt.image_width is not None
        opt.tgt_size = (opt.image_height, opt.image_width)
    else:
        opt.tgt_size = None
    opt.out = os.path.join(opt.out_dir, opt.name)
    opt.tb_dir = os.path.join(opt.out_dir, 'tensorboard_out', opt.name)
    if opt.load_weights is not None:
        opt.load_weights_path = os.path.join(opt.out_dir, opt.load_weights, 'checkpoint.pth.tar')
        assert os.path.isfile(opt.load_weights_path), opt.load_weights_path
    elif opt.resume:
        opt.load_weights_path = os.path.join(opt.out_dir, f'{opt.name}', 'checkpoint.pth.tar')
        assert os.path.isfile(opt.load_weights_path), opt.load_weights_path
    assert math.isclose(opt.body_rays_ratio + opt.border_rays_ratio + opt.bkg_rays_ratio, 1.0), f'{opt.body_rays_ratio + opt.border_rays_ratio + opt.bkg_rays_ratio}'

    if hasattr(opt, 'ablate_nerft') and opt.ablate_nerft:
        assert opt.raw_pos_dim == 4
        assert opt.train_mode == 'bkg'


    #assert opt.normalize == True

    options.print_opt(opt)
    options.save_opt(opt)
    if opt.train_mode == 'bkg':
        train_background(opt)
    elif opt.train_mode in ['smpl_only', 'smpl_and_offset']:
        train_human(opt)
    else:
        raise ValueError
