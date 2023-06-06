#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import argparse
import math
import numpy as np
import os
import random
import torch
from torch.utils import data

from data_io import cache_helper
from data_io import neuman_helper
from options import options
from options.options import str2bool
from utils import ray_utils
from utils.constant import PATCH_SIZE, PATCH_SIZE_SQUARED, NEAR_INDEX, FAR_INDEX, TRAIN_SET_LENGTH, \
    VALIDATION_SET_LENGTH


class ImagesOnlyDataset(data.Dataset):
    '''
    We use random rays from a SINGLE time stamp per batch.
    '''

    def __init__(self, opt, scene, dset_type, split, near_far_cache=None):
        '''
        Args:
            opt (Namespace): options
            scene (BaseScene): the background scene
            dset_type (str): train/val/test set
            split (str): split file path
            near_far_cache (dict, optional): The SMPL mesh guided near and far for each pixels. Defaults to None.
                                             Key: file name of image.
                                             Value: ndarray with shape [h, w, 3].
        '''
        self.opt = opt
        self.scene = scene
        self.device = scene.captures[0].posed_mesh.device
        self.dset_type = dset_type
        self.split = split
        self.inclusions = neuman_helper.read_text(split)
        print(f'{dset_type} dataset has {len(self.inclusions)} samples: {self.inclusions}')
        self.white_bkg = opt.white_bkg
        self.batch_size = opt.rays_per_batch
        if near_far_cache is None:
            cache_helper.export_near_far_cache(opt, scene, opt.geo_threshold, opt.chunk, self.device)
            self.near_far_cache = cache_helper.load_near_far_cache(opt, scene, opt.geo_threshold)
        else:
            self.near_far_cache = near_far_cache

        # print(self.near_far_cache)
        # exit(0)
        self.num_patch = 1 if opt.penalize_lpips > 0 else 0
        # self.cap_id = None

    def __len__(self):
        return len(self.inclusions)


    # def get_num_rays_dict_patch(self, num):


    def __getitem__(self, index):
        '''
        NeRF requires 4K+ rays per gradient decent, so we will return the ray batch directly.
        '''
        # if self.cap_id is None:
        #     cap_id = self.scene.fname_to_index_dict[random.choice(self.inclusions)]
        # else:
        #     cap_id = self.cap_id

        # print(self.inclusions[index])
        # exit()

        # TODO rewrite to get the capture specified by cap_id, instead of randomly sampling one
        cap_id = self.scene.fname_to_index_dict[
            self.inclusions[index]
        ]
        caps = self.scene.get_captures_by_view_id(cap_id)
        assert len(caps) == 1, 'one camera per one iteration'

        cap = caps[0]

        # print(self.near_far_cache[os.path.basename(cap.image_path)].shape, self.near_far_cache[os.path.basename(cap.image_path)])
        # exit()
        # print(cap.image.shape)
        out = {
            'image': torch.tensor(cap.image, dtype=torch.int32),
            'mask': torch.tensor(cap.mask, dtype=torch.int32),
            'border_mask': torch.tensor(cap.border_mask, dtype=torch.int32),
            'binary_mask': torch.tensor(cap.binary_mask, dtype=torch.int32),
            'near_far_cache': torch.tensor(self.near_far_cache[os.path.basename(cap.image_path)], dtype=torch.float32),
            'image_path': cap.image_path,

            'near_human': torch.tensor(cap.near['human'], dtype=torch.float32),
            'far_human': torch.tensor(cap.far['human'], dtype=torch.float32),

            # near/far for background is hardcoded
            'near_bkg': torch.tensor(-1000.0, dtype=torch.float32),
            'far_bkg': torch.tensor(1000.0, dtype=torch.float32),

            'intrinsic_matrix': torch.tensor(cap.intrinsic_matrix, dtype=torch.float32),
            'cam2world': torch.tensor(cap.cam_pose.camera_to_world, dtype=torch.float32),
            'camera_center_in_world': torch.tensor(cap.cam_pose.camera_center_in_world, dtype=torch.float32),
            'cur_view_f':    torch.tensor(cap.frame_id['frame_id'] / cap.frame_id['total_frames'], dtype=torch.float32),
            'cur_view':      torch.tensor(cap.frame_id['frame_id'], dtype=torch.int32),
            'cap_id':        torch.tensor(cap_id, dtype=torch.int32),
        }
        return out
