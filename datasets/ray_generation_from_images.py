import os
import random

import torch
from torch.utils import data

from data_io import cache_helper
from utils import ray_utils
from utils.constant import PATCH_SIZE, PATCH_SIZE_SQUARED, NEAR_INDEX, FAR_INDEX, TRAIN_SET_LENGTH, VALIDATION_SET_LENGTH
from data_io import neuman_helper

import argparse
from options.options import str2bool
from options import options
import math

import numpy as np

def get_left_upper_corner(img, pos, size=PATCH_SIZE):
    '''
    pos - [x, y]
    return the left upper corner of a patch centered(as centered as possible) at pos.
    '''
    # print(img.shape)
    h, w, _ = img.shape
    lu_y = int(pos[1] - size // 2)
    lu_x = int(pos[0] - size // 2)
    if lu_y < 0:
        lu_y -= lu_y
    if lu_x < 0:
        lu_x -= lu_x
    if lu_y + size > h:
        lu_y -= (lu_y + size) - h
    if lu_x + size > w:
        lu_x -= (lu_x + size) - w
    return torch.tensor([lu_x, lu_y], device=img.device)



class RaysFromImagesGenerator:
    '''
    We use random rays from a SINGLE time stamp per batch.
    '''

    def __init__(self, opt):
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
        self.white_bkg = opt.white_bkg
        self.batch_size = opt.rays_per_batch
        self.num_patch = 1 if opt.penalize_lpips > 0 else 0

    def get_num_rays_dict(self, num):
        """
        Return the number of rays for each type of rays.

        Args:
            num (int): number of rays in total.

        Returns:
            dict: number of rays for each type of rays.
            {
            'num_body_rays': num_body_rays,
            'num_border_rays': num_border_rays,
            'num_bkg_rays': num_bkg_rays
            }
        """
        num_body_rays = int(round(num * self.opt.body_rays_ratio))
        num_border_rays = int(round(num * self.opt.border_rays_ratio)) if self.opt.dilation > 0 else 0
        num_bkg_rays = int(round(num * self.opt.bkg_rays_ratio))

        leftover = num - num_body_rays - num_border_rays - num_bkg_rays
        num_body_rays += leftover

        # arr = np.array([num_body_rays, num_border_rays, num_bkg_rays])
        # arr[arr.argmax()] += leftover
        # num_body_rays, num_border_rays, num_bkg_rays = arr

        assert min(num_body_rays, num_border_rays,
                   num_bkg_rays) >= 0, f'{min(num_body_rays, num_border_rays, num_bkg_rays)}'
        assert num_body_rays + num_bkg_rays + num_border_rays == num, f'Total number of rays {num} does not match the sum of body rays {num_body_rays}, border rays {num_border_rays} and bkg rays {num_bkg_rays}'

        return {
            'num_body_rays': num_body_rays,
            'num_border_rays': num_border_rays,
            'num_bkg_rays': num_bkg_rays,
        }


    def generate_rays_from_images(self, batch_img):

        device = batch_img['image'].device

        # print('running on device: ', device)
        # print(batch_img)
        #exit()

        # make bins for lpips loss
        if self.num_patch == 0:
            # do not penalize lpips, not sampling patch
            bins = [self.batch_size]
        elif self.num_patch == 1:
            # penalize lpips, sampling patch
            # first part of the batch is the sampled patch
            # random rays for the leftover
            assert self.batch_size > PATCH_SIZE_SQUARED
            bins = [PATCH_SIZE_SQUARED, self.batch_size - PATCH_SIZE_SQUARED]
            # caps = [caps[0], caps[0]]
        else:
            raise ValueError('only support 1 patch')

        # randomly disable lpips loss
        if random.random() < self.opt.body_rays_ratio:
            need_patch = True
        else:
            need_patch = False
        patch_counter = 0

        # assert len(caps) == len(bins), f'{len(caps)} != {len(bins)}'

        colors_list = torch.tensor([], device=device, dtype=torch.float32)
        orig_list = torch.tensor([], device=device, dtype=torch.float32)
        dir_list = torch.tensor([], device=device, dtype=torch.float32)
        human_near_list = torch.tensor([], device=device, dtype=torch.float32)
        human_far_list = torch.tensor([], device=device, dtype=torch.float32)
        bkg_near_list = torch.tensor([], device=device, dtype=torch.float32)
        bkg_far_list = torch.tensor([], device=device, dtype=torch.float32)
        is_bkg_list = torch.tensor([], device=device, dtype=torch.int32)
        is_hit_list = torch.tensor([], device=device, dtype=torch.int32)
        # coords_list = {i: torch.tensor([], device=device) for i in range(len(bins))}


        for cam_id, num in enumerate(bins):
            if num == 0:
                continue

            img = batch_img['image']

            # select which rays to sample: patch rays for lpips loss or regular rays for rendering
            if self.num_patch == 1 and need_patch and patch_counter == 0:
                # sample patch rays
                assert num == PATCH_SIZE_SQUARED
                num_rays_dict = {'num_patch_rays': num}
                patch_counter += 1
            else:
                # sample body, border, bkg rays
                num_rays_dict = self.get_num_rays_dict(num)

            # sample rays for each type of rays:
            # either 'body', 'border' or 'bkg' for regular rays or 'num_patch_rays' for lpips loss
            for ray_key, num_rays in num_rays_dict.items():
                if num_rays == 0:
                    continue

                # get ray coordinates
                if ray_key == 'num_body_rays':
                    coords = torch.argwhere(batch_img['mask'] != 0)
                elif ray_key == 'num_border_rays':
                    coords = torch.argwhere(batch_img['border_mask'] == 1)
                elif ray_key == 'num_bkg_rays':
                    coords = torch.argwhere(batch_img['mask'] == 0)

                elif ray_key == 'num_patch_rays':
                    nonzero_mask = torch.argwhere(batch_img['mask'] != 0)

                    seed = nonzero_mask[
                        torch.randint(0, nonzero_mask.shape[0], (1,))
                    ].squeeze()

                    seed = torch.flip(
                        get_left_upper_corner(img, torch.flip(seed, [0])), #[::-1],
                        [0]
                    )

                    bound = seed + PATCH_SIZE

                    # A hacky way to obtain the coords
                    # Assuming argwhere returns in order
                    temp = torch.zeros_like(batch_img['mask'])
                    assert temp[seed[0]:bound[0], seed[1]:bound[1]].shape == (PATCH_SIZE, PATCH_SIZE), 'wrong patch size'

                    temp[seed[0]:bound[0], seed[1]:bound[1]] = 1

                    coords = torch.argwhere(temp == 1).int()
                    # An alternative which does not use argwhere
                    # y, x = np.meshgrid(
                    #                 np.linspace(seed[1], bound[1]-1, num=PATCH_SIZE),
                    #                 np.linspace(seed[0], bound[0]-1, num=PATCH_SIZE),
                    #         ).astype(int)
                    # coords = np.stack([x, y], -1).reshape(-1, 2)
                    check = (img[seed[0]:bound[0], seed[1]:bound[1]] / 255).float()
                else:
                    raise ValueError

                # print('coords', coords.shape, coords.dtype)
                # rearrange coords to (y, x)
                if ray_key == 'num_patch_rays':
                    coords = torch.flip(coords, [1])
                else:
                    coords = torch.flip(coords[
                                 torch.randperm(len(coords))[:num_rays]
                             ],
                                        [1]
                                        )
                # # TODO is this necessary?
                # coords_list[cam_id].append(coords)

                # print('coords', coords.shape, coords)
                # exit()
                # get ray colors
                colors = (img[coords[:, 1], coords[:, 0]] / 255).float()

                # check if rays are forming a patch
                if ray_key == 'num_patch_rays':
                    assert (colors.reshape(PATCH_SIZE, PATCH_SIZE, -1) == check).all(), 'rays not forming a patch'

                # check if rays are hitting the background or human
                is_bkg = 1 - batch_img['binary_mask'][coords[:, 1], coords[:, 0]]

                # get ray origins and directions
                orig, dirs = ray_utils.shot_rays_torch(
                    xys=coords,
                    intrinsic_matrix=batch_img['intrinsic_matrix'],
                    camera_to_world=batch_img['cam2world'],
                    camera_center_in_world=batch_img['camera_center_in_world'],
                )

                # I am not sure what this is for
                cache = batch_img['near_far_cache'][coords[:, 1], coords[:, 0]]
                valid = cache[..., NEAR_INDEX] < cache[..., FAR_INDEX]

                # copy near/far to all rays
                human_far = batch_img['far_human'].repeat(
                    num_rays, 1
                )
                human_near = batch_img['near_human'].repeat(
                    num_rays, 1
                )

                # I am not sure what this is for
                human_near[valid, 0] = cache[valid][:, NEAR_INDEX]
                human_far[valid, 0] = cache[valid][:, FAR_INDEX]

                # copy near/far to all rays
                bkg_far = batch_img['far_bkg'].repeat(
                    num_rays, 1
                )
                bkg_near = batch_img['near_bkg'].repeat(
                    num_rays, 1
                )
                assert ((human_near <= human_far).all())

                colors_list = torch.cat((colors_list, colors), 0)
                orig_list = torch.cat((orig_list, orig), 0)
                dir_list = torch.cat((dir_list, dirs), 0)
                human_near_list = torch.cat((human_near_list, human_near), 0)
                human_far_list = torch.cat((human_far_list, human_far), 0)
                bkg_near_list = torch.cat((bkg_near_list, bkg_near), 0)
                bkg_far_list = torch.cat((bkg_far_list, bkg_far), 0)
                is_bkg_list = torch.cat((is_bkg_list, is_bkg.int()), 0)
                is_hit_list = torch.cat((is_hit_list, valid.int()), 0)

        # print('colors_list', colors_list.shape, colors_list.dtype)
        # print('orig_list', orig_list.shape, orig_list.dtype)
        # print('dir_list', dir_list.shape, dir_list.dtype)
        # print('human_near_list', human_near_list.shape, human_near_list.dtype)
        # print('human_far_list', human_far_list.shape, human_far_list.dtype)
        # print('bkg_near_list', bkg_near_list.shape, bkg_near_list.dtype)
        # print('bkg_far_list', bkg_far_list.shape, bkg_far_list.dtype)
        # print('is_bkg_list', is_bkg_list.shape, is_bkg_list.dtype)
        # print('is_hit_list', is_hit_list.shape, is_hit_list.dtype)

        # assert len(coords_list) == len(bins), f'{len(coords_list)} != {len(bins)}'
        assert colors_list.shape[0] == \
               orig_list.shape[0] == \
               dir_list.shape[0] == \
               human_near_list.shape[0] == \
               human_far_list.shape[0] == \
               bkg_near_list.shape[0] == \
               bkg_far_list.shape[0] == \
               is_bkg_list.shape[0] == \
               is_hit_list.shape[0] == \
               self.batch_size


        out = {
            'color': colors_list,
            'origin': orig_list,
            'direction': dir_list,
            'human_near': human_near_list,
            'human_far': human_far_list,
            'bkg_near': bkg_near_list,
            'bkg_far': bkg_far_list,
            'is_bkg': is_bkg_list,
            'is_hit': is_hit_list,

            'cur_view_f': batch_img['cur_view_f'],
            'cur_view': batch_img['cur_view'],
            'cap_id': batch_img['cap_id'],

            'patch_counter': torch.tensor(patch_counter, dtype=torch.int32, device=colors_list.device),
        }

        # print(
        #     'cur_view_f', type((cap.frame_id['frame_id'] / cap.frame_id['total_frames'])),
        #     'cur_view', type(cap.frame_id['frame_id']),
        #     'cap_id', type(cap_id),
        # )
        return out