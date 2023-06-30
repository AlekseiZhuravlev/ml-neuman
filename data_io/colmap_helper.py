# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/datasets/colmap_helper.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


import os
import re
from collections import namedtuple

import numpy as np
from tqdm import tqdm

from geometry.basics import Translation, Rotation
from cameras.camera_pose import CameraPose
from cameras.pinhole_camera import PinholeCamera
from cameras import captures as captures_module
from scenes import scene as scene_module

import json
import logging


ImageMeta = namedtuple('ImageMeta', ['image_id', 'camera_pose', 'camera_id', 'image_path', 'old_image_name'])


class ColmapAsciiReader():
    def __init__(self):
        pass

    @classmethod
    def read_scene(cls, scene_dir, images_dir, tgt_size=None, order='default'):
        # point_cloud_path = os.path.join(scene_dir, 'points3D.txt')
        cameras_path = os.path.join(scene_dir, 'cameras.txt')
        images_path = os.path.join(scene_dir, 'images.txt')
        captures = cls.read_captures(images_path, cameras_path, images_dir, tgt_size, order)
        # point_cloud = cls.read_point_cloud(point_cloud_path)
        scene = scene_module.ImageFileScene(captures, point_cloud=None)
        return scene

    @staticmethod
    def read_point_cloud(points_txt_path):
        with open(points_txt_path, "r") as fid:
            line = fid.readline()
            assert line == '# 3D point list with one line of data per point:\n'
            line = fid.readline()
            assert line == '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n'
            line = fid.readline()
            assert re.search('^# Number of points: \d+, mean track length: [-+]?\d*\.\d+|\d+\n$', line)
            num_points, mean_track_length = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            num_points = int(num_points)
            mean_track_length = float(mean_track_length)

            xyz = np.zeros((num_points, 3), dtype=np.float32)
            rgb = np.zeros((num_points, 3), dtype=np.float32)

            for i in tqdm(range(num_points), desc='reading point cloud'):
                elems = fid.readline().split()
                xyz[i] = list(map(float, elems[1:4]))
                rgb[i] = list(map(float, elems[4:7]))
            pcd = np.concatenate([xyz, rgb], axis=1)
        return pcd

    @classmethod
    def read_captures(cls, images_txt_path, cameras_txt_path, images_dir, tgt_size, order='default'):
        captures = []
        cameras, images_meta = cls.read_cameras_and_meta(cameras_txt_path)

        print('cameras', len(cameras))
        print('images_meta', len(images_meta))

        # images_meta = cls.read_images_meta(images_txt_path, images_dir)
        if order == 'default':
            keys = images_meta.keys()
        elif order == 'video':
            keys = []
            frames = []
            for k, v in images_meta.items():
                keys.append(k)
                frames.append(os.path.basename(v.image_path))
            keys = [x for _, x in sorted(zip(frames, keys))]
        else:
            raise ValueError(f'unknown order: {order}')
        for i, key in enumerate(keys):
            cur_cam_id = images_meta[key].camera_id
            cur_cam = cameras[cur_cam_id]
            cur_camera_pose = images_meta[key].camera_pose
            cur_image_path = images_meta[key].image_path
            cur_image_old_name = images_meta[key].old_image_name

            # print(cur_cam_id, cur_cam, cur_camera_pose, cur_image_path, cur_image_old_name)
            if tgt_size is None:
                cap = captures_module.RGBPinholeCapture(cur_image_path, cur_cam, cur_camera_pose, cur_image_old_name)
            else:
                cap = captures_module.ResizedRGBPinholeCapture(cur_image_path, cur_cam, cur_camera_pose, tgt_size)
            if order == 'video':
                cap.frame_id = {'frame_id': i, 'total_frames': len(images_meta)}
            captures.append(cap)
        return captures

    @classmethod
    def read_cameras_and_meta(cls, cameras_txt_path):
        # with open(self.base_folder + '/annotations/' + self.split + '/InterHand2.6M_' + split + '_camera.json') as f:
        with open(
                '/itet-stor/azhuavlev/net_scratch/Projects/Data/Interhand_masked/annotations/test/InterHand2.6M_test_camera.json',
                'r') as f:
            camera_params_dict = json.load(f)

        # camera_ids = ['400262', '400263', '400264', '400265', '400284']
        capture_n = '0'
        data_path = '/home/azhuavlev/Desktop/Data/InterHand_Neuman/02/'

        with open(data_path+'mapping.json', 'r') as f:
            cam_img_mapping = json.load(f)

        cameras = {}
        images_meta = {}

        # print(json.dumps(cam_img_mapping, indent=4, sort_keys=True))
        # exit()

        for new_camera_id in sorted(cam_img_mapping.keys()):
            # print(type(camera_id))
            # exit()

            old_camera_id = cam_img_mapping[str(new_camera_id)]['old_camera_id']
            new_camera_id = int(new_camera_id)


            # load camera parameters
            # campos = camera_params_dict[capture_n]['campos'][old_camera_id]
            # campos = np.array(campos, dtype=np.float32) / 1000.0 # convert to meters
            # camrot = camera_params_dict[capture_n]['camrot'][old_camera_id]
            focal = camera_params_dict[capture_n]['focal'][old_camera_id] # for x and y
            princpt = camera_params_dict[capture_n]['princpt'][old_camera_id] # for x and y

            t_vector = np.array(camera_params_dict[capture_n]['campos'][old_camera_id], dtype=np.float32).reshape(3) / 1000
            R_mat = np.array(camera_params_dict[capture_n]['camrot'][old_camera_id], dtype=np.float32).reshape(3, 3)
            t_vector = -np.dot(R_mat, t_vector.reshape(3, 1)).reshape(3)

            # print('campos', campos)
            # print('camrot', camrot)
            # print('t', t)
            # print('R', R)
            # exit()
            # get image size
            width, height = 334, 512

            cur_cam = PinholeCamera(width, height, focal[0], focal[1], princpt[0], princpt[1])
            # logging.debug(f'camera {camera_id}:\n{cur_cam}')
            cameras[new_camera_id] = cur_cam

            # camera_folder = pose_path + '/' + f'cam{camera_id}'
            img_list = cam_img_mapping[str(new_camera_id)]['images_list']

            for img_dict in img_list:
                old_img_name = img_dict['old_img_name']
                new_img_name = img_dict['new_img_name']

                image_path = data_path + 'images/' + new_img_name

                t = Translation(t_vector)

                # call Rotation.from_matrix to convert the rotation matrix to a quaternion, pass camrot (2d list) as ndarray
                r = Rotation.from_matrix(R_mat)
                camera_pose = CameraPose(t, r)
                # logging.debug(f'camera pose:\n{camera_pose}')

                image_id = int(f"{new_img_name[:-4]}")
                images_meta[image_id] = ImageMeta(image_id, camera_pose, new_camera_id, image_path, old_img_name)
                # print(images_meta[image_id], images_meta[image_id].camera_pose)
                # print(f"{new_img_name[:-3]}", camera_pose, camera_id, image_path, old_img_name)


        # print(cameras)
        return cameras, images_meta


    @classmethod
    def read_images_meta(cls, images_txt_path, images_dir):
        images_meta = {}
        with open(images_txt_path, "r") as fid:
            line = fid.readline()
            assert line == '# Image list with two lines of data per image:\n'
            line = fid.readline()
            assert line == '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n'
            line = fid.readline()
            assert line == '#   POINTS2D[] as (X, Y, POINT3D_ID)\n'
            line = fid.readline()
            assert re.search('^# Number of images: \d+, mean observations per image: [-+]?\d*\.\d+|\d+\n$', line)
            num_images, mean_ob_per_img = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            num_images = int(num_images)
            mean_ob_per_img = float(mean_ob_per_img)

            for _ in tqdm(range(num_images), desc='reading images meta'):
                elems = fid.readline().split()
                assert len(elems) == 10
                line = fid.readline()
                image_path = os.path.join(images_dir, elems[9])
                assert os.path.isfile(image_path), f'missing file: {image_path}'
                image_id = int(elems[0])
                qw, qx, qy, qz, tx, ty, tz = list(map(float, elems[1:8]))
                t = Translation(np.array([tx, ty, tz], dtype=np.float32))
                r = Rotation(np.array([qw, qx, qy, qz], dtype=np.float32))
                camera_pose = CameraPose(t, r)
                camera_id = int(elems[8])
                assert image_id not in images_meta, f'duplicated image, id: {image_id}, path: {image_path}'
                images_meta[image_id] = ImageMeta(image_id, camera_pose, camera_id, image_path)
        return images_meta

if __name__=='__main__':
    ColmapAsciiReader.read_captures('1', '2', 's', None, 'video')