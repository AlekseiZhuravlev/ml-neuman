import sys, os
import glob
sys.path.append("/home/azhuavlev/PycharmProjects/ml-neuman_mano")


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2
from mano_custom import mano_pytorch3d
from scipy import ndimage
from pytorch3d.transforms.so3 import so3_exponential_map, so3_log_map
from losses.canonical_utils.cameras_canonical import get_look_at_view_R_T
from losses.canonical_utils.render_canonical import RendererCanonical
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
)


class NeumanDataset(torch.utils.data.Dataset):
    def __init__(self, exp_dir, cap_ids):
        self.exp_dir = exp_dir
        self.cap_ids = cap_ids
        self.hand_model = mano_pytorch3d.create_mano_custom(return_right_hand=False)

        self.load_cameras_interhand()
        self.load_silhouettes()
        self.load_images()
        self.load_mano()

        self.mask_images()
        self.create_zero_pose_silhouettes()



    def load_cameras_interhand(self):
        cam_path = self.exp_dir + '/cameras'

        self.camera_params_interhand = []
        self.camera_params_training = []

        self.Rs_can, self.Ts_can = get_look_at_view_R_T(len(self.cap_ids), random_cameras=False, device='cpu')

        for j_index, i in enumerate(self.cap_ids):
            json_file = cam_path + f'/{i:05d}.json'
            with open(json_file) as f:
                data = json.load(f)

                campos = np.array(data['campos'], dtype=np.float32).reshape(3) / 1000.0
                camrot = np.array(data['camrot'], dtype=np.float32).reshape(3, 3)
                focal = np.array(data['focal'], dtype=np.float32).reshape(2)
                princpt = np.array(data['princpt'], dtype=np.float32).reshape(2)

                R = camrot
                t = -np.dot(camrot, campos.reshape(3, 1)).reshape(3)  # -Rt -> t

                intrinsic_mat = np.array([
                    [focal[0], 0, princpt[0]],
                    [0, focal[1], princpt[1]],
                    [0, 0, 1]
                ], dtype=np.float32)

                # (height, width)
                image_size = np.array([512.0, 334.0], dtype=np.float32)

                self.camera_params_interhand.append({
                    'R': R,
                    't': t,
                    'intrinsic_mat': intrinsic_mat,
                    'image_size': image_size,
                    'focal': focal,
                    'princpt': princpt,
                    'campos': campos,
                    'camrot': camrot,
                })

                R_train = torch.eye(3)
                t_train = torch.zeros(3)
                self.camera_params_training.append({
                    'R_pytorch3d': R_train,
                    't_pytorch3d': t_train,
                    'R_can': self.Rs_can[j_index],
                    't_can': self.Ts_can[j_index],
                    'intrinsic_mat': intrinsic_mat,
                    'image_size': image_size,
                    'focal': focal,
                    'princpt': princpt,
                })
#
    def load_images(self):
        img_path = self.exp_dir + '/images'

        self.images = []

        for i in self.cap_ids:
            img_file = img_path + f'/{i:05d}.png'
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

            self.images.append(img)

    def load_silhouettes(self):
        sil_path = self.exp_dir + '/segmentations'

        self.silhouettes = []

        for i in self.cap_ids:
            sil_file = sil_path + f'/{i:05d}.png'
            sil = cv2.imread(sil_file, cv2.IMREAD_GRAYSCALE)
            sil = sil.astype(np.float32) / 255.0

            # invert silhouettes
            sil = 1.0 - sil
            sil = sil.clip(0.0, 1.0)

            # TODO added dilation
            # mask_dilated_5 = ndimage.binary_dilation(mask, iterations=5) - mask.numpy()
            sil_dilated_10 = ndimage.binary_dilation(sil, iterations=10).astype(np.float32).clip(0.0, 1.0)

            self.silhouettes.append(
                sil_dilated_10
            )

    def load_mano(self):
        mano_path = self.exp_dir + '/mano'
        joints_path = self.exp_dir + '/joints'

        hand_model = self.hand_model

        self.manos = []
        for j_curr_item, i_cap_id in enumerate(self.cap_ids):
            # print('j_curr_item', j_curr_item, 'i_cap_id', i_cap_id)

            mano_file = mano_path + f'/{i_cap_id:05d}.json'
            with open(mano_file) as f:
                mano = json.load(f)['left']

            root_pose, hand_pose, shape, trans = self.apply_extr_to_mano(mano, hand_model, j_curr_item)

            verts_zero_pose_xyz, Ts_xyz = hand_model.verts_transformations_xyz(
                betas=shape,
                global_orient=root_pose,
                hand_pose=hand_pose,
                transl=trans
            )
            verts_zero_pose_py3d = torch.stack(
                (-verts_zero_pose_xyz[:, :, 0],
                 -verts_zero_pose_xyz[:, :, 1],
                 verts_zero_pose_xyz[:, :, 2]),
                2)
            vertices_py3d = hand_model.forward_pytorch3d(
                betas=shape,
                global_orient=root_pose,
                hand_pose=hand_pose,
                transl=trans
            )

            # # check if joints_path exists
            # if os.path.exists(joints_path):
            #     joints_file = joints_path + f'/{i_cap_id:05d}.json'
            #     with open(joints_file) as f:
            #         joints = json.load(f)
            # else:
            #     joints = 0

            mano_dict = {
                'root_pose': root_pose.squeeze(0),
                'hand_pose': hand_pose.squeeze(0),
                'shape': shape.squeeze(0),
                'trans':trans.squeeze(0),
                'verts': vertices_py3d.squeeze(0),
                'verts_zero': verts_zero_pose_py3d.squeeze(0),
                'Ts': Ts_xyz.squeeze(0),
                # 'joints': joints,
            }
            self.manos.append(mano_dict)

    def apply_extr_to_mano(self, mano_param, hand_model, j_curr_item):

        R = self.camera_params_interhand[j_curr_item]['R']
        t = self.camera_params_interhand[j_curr_item]['t']

        mano_pose = torch.tensor(mano_param['pose'], dtype=torch.float32).view(-1, 3)
        root_pose = mano_pose[0].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))  # multiply camera rotation to MANO root pose
        root_pose = torch.from_numpy(root_pose).view(1, 3)
        hand_pose = mano_pose[1:, :].view(1, -1)
        shape = torch.tensor(mano_param['shape'], dtype=torch.float32).view(1, -1)

        with torch.no_grad():
            output = hand_model(global_orient=root_pose, hand_pose=hand_pose,
                                           betas=shape)  # this is rotation-aligned, but not translation-aligned with the camera coordinates
        mesh = output.vertices[0].detach().numpy()

        joint_regressor = np.load(
            '/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/camera_conversion/J_regressor_mano_ih26m.npy')
        joint_from_mesh = np.dot(joint_regressor, mesh)

        # compenstate rotation (translation from origin to root joint was not cancled)
        root_joint_idx = 20
        root_joint = joint_from_mesh[root_joint_idx, None, :]

        # change translation vector
        trans = np.array(mano_param['trans'])
        trans = np.dot(R, trans.reshape(3, 1)).reshape(1, 3) - root_joint + np.dot(R, root_joint.transpose(1,
                                                                                                           0)).transpose(
            1, 0) + t
        trans = torch.tensor(trans, dtype=torch.float32).view(1, 3)

        return root_pose, hand_pose, shape, trans

    def mask_images(self):
        for i in range(len(self.images)):
            silh_3ch = np.stack([self.silhouettes[i]] * 3, axis=2).astype(np.float32)
            self.images[i] = self.images[i] * silh_3ch

    def create_zero_pose_silhouettes(self):

        device='cuda:0' if torch.cuda.is_available() else 'cpu'
        cameras = FoVPerspectiveCameras(
            R=self.Rs_can,
            T=self.Ts_can,
            znear=0.01,
            zfar=10,
            device=device,
            # image_size=image_sizes
        )

        renderer = RendererCanonical(cameras)

        with torch.no_grad():
            self.silhouettes_zero_pose = renderer.render_zero_pose_sil(
                self.manos[0]['verts_zero'].to(device),
                torch.from_numpy(self.hand_model.faces.astype(np.int32))[None, :, :].to(device)
            )
        self.silhouettes_zero_pose = self.silhouettes_zero_pose.cpu().squeeze(-1)
        print('silhouettes_zero_pose.shape', self.silhouettes_zero_pose.shape)
        # plot one silhouette
        # import matplotlib.pyplot as plt
        #
        # for i in range(len(self.silhouettes_zero_pose)):
        #     plt.figure()
        #     plt.imshow(self.silhouettes_zero_pose[i].squeeze().cpu().numpy(), cmap='gray')
        #     plt.savefig(f'/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/losses/canonical_utils/images/silhouette_zero_pose_{i}.png')
        #     plt.close()
        # exit()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.camera_params_training[idx], self.images[idx], self.silhouettes[idx], self.silhouettes_zero_pose[idx], self.manos[idx]