import sys, os
import glob
sys.path.append("/home/azhuavlev/PycharmProjects/ml-neuman_mano")


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2
import mano_pytorch3d
from scipy import ndimage
from pytorch3d.transforms.so3 import so3_exponential_map, so3_log_map



class NeumanDataset(torch.utils.data.Dataset):
    def __init__(self, exp_dir, cap_ids):
        self.exp_dir = exp_dir
        self.cap_ids = cap_ids

        self.load_cameras()
        self.load_silhouettes()
        self.load_images()
        self.load_mano()

        self.mask_images()


    def load_cameras(self):
        cam_path = self.exp_dir + '/cameras'

        self.camera_params_opencv = []
        for i in self.cap_ids:
            json_file = cam_path + f'/{i:05d}.json'
            with open(json_file) as f:
                data = json.load(f)

                campos = np.array(data['campos'], dtype=np.float32).reshape(3) / 1000.0
                camrot = np.array(data['camrot'], dtype=np.float32).reshape(3, 3)
                focal = np.array(data['focal'], dtype=np.float32).reshape(2)
                princpt = np.array(data['princpt'], dtype=np.float32).reshape(2)

                R = camrot
                t = -np.dot(camrot, campos.reshape(3, 1)).reshape(3)  # -Rt -> t

                # convert rotation matrix to rotation vector and flip the sign of x and y
                R = torch.tensor(R).unsqueeze(0)
                rot_vec = so3_log_map(R)
                rot_vec[:, :2] *= -1

                # convert back to rotation matrix
                R_pytorch3d = so3_exponential_map(rot_vec).squeeze(0).transpose(1, 0)

                # flip the sign of x and y for translation vector
                t_pytorch3d = t.copy()
                t_pytorch3d[:2] *= -1

                # R = torch.tensor(R).unsqueeze(0)
                #
                # rot_vec = so3_log_map(R)
                # rot_vec_adj = rot_vec.clone()
                # rot_vec_adj[:, :2] *= -1
                #
                # R = so3_exponential_map(rot_vec_adj)
                # R = R.squeeze(0).numpy()
                #
                # R_pytorch3d = R.copy().transpose(1, 0)
                # T_pytorch3d = t.copy()
                # T_pytorch3d[:2] *= -1
                #
                # R_pytorch3d = torch.tensor(R_pytorch3d)#.unsqueeze(0)
                # t_pytorch3d = torch.tensor(T_pytorch3d)#.unsqueeze(0)

                intrinsic_mat = np.array([
                    [focal[0], 0, princpt[0]],
                    [0, focal[1], princpt[1]],
                    [0, 0, 1]
                ], dtype=np.float32)

                # (height, width)
                image_size = np.array([512.0, 334.0], dtype=np.float32)

                self.camera_params_opencv.append({
                    'R_pytorch3d': R_pytorch3d,
                    't_pytorch3d': t_pytorch3d,
                    'intrinsic_mat': intrinsic_mat,
                    'image_size': image_size,
                    'focal': focal,
                    'princpt': princpt,
                    'campos': campos,
                    'camrot': camrot,
                })
#
    def load_images(self):
        img_path = self.exp_dir + '/images'

        self.images = []

        for i, indx in enumerate(self.cap_ids):
            img_file = img_path + f'/{i:05d}.png'
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

            self.images.append(img)

    def load_silhouettes(self):
        sil_path = self.exp_dir + '/segmentations'

        self.silhouettes = []

        for i, indx in enumerate(self.cap_ids):
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

        hand_model = mano_pytorch3d.MANOCustom(
            model_path='/home/azhuavlev/Desktop/Data/models/mano/MANO_LEFT.pkl',
            is_rhand=False,
            use_pca=False,
        )

        self.manos = []
        for i in self.cap_ids:
            mano_file = mano_path + f'/{i:05d}.json'
            with open(mano_file) as f:
                mano = json.load(f)['left']

            # get mano pose and reshape it
            mano_pose = torch.tensor(mano['pose'], dtype=torch.float32).view(-1, 3)

            # root pose is at position 0, pose of rest of the hand is at positions [1:]
            root_pose = mano_pose[0]
            hand_pose = mano_pose[1:, :].view(-1)

            # get betas (called shapes here) and translation vector
            shape = torch.tensor(mano['shape'], dtype=torch.float32)
            trans = torch.tensor(mano['trans'], dtype=torch.float32)

            _, Ts_xyz = hand_model.verts_transformations_pytorch3d(
                betas=shape.unsqueeze(0),
                global_orient=root_pose.unsqueeze(0),
                hand_pose=hand_pose.unsqueeze(0),
                transl=trans.unsqueeze(0)
            )

            vertices_py3d = hand_model.forward_pytorch3d(
                betas=shape.unsqueeze(0),
                global_orient=root_pose.unsqueeze(0),
                hand_pose=hand_pose.unsqueeze(0),
                transl=trans.unsqueeze(0)
            )

            # check if joints_path exists
            if os.path.exists(joints_path):
                joints_file = joints_path + f'/{i:05d}.json'
                with open(joints_file) as f:
                    joints = json.load(f)
            else:
                joints = 0


            mano_dict = {
                'root_pose': root_pose,
                'hand_pose': hand_pose,
                'pose': mano_pose,
                'shape': shape,
                'trans':trans,
                'verts': vertices_py3d.squeeze(0),
                'Ts': Ts_xyz.squeeze(0),
                'joints': joints,
            }

            self.manos.append(mano_dict)

    def mask_images(self):
        for i in range(len(self.images)):
            silh_3ch = np.stack([self.silhouettes[i]] * 3, axis=2).astype(np.float32)
            self.images[i] = self.images[i] * silh_3ch


    def __len__(self):
        return len(self.camera_params_opencv)

    def __getitem__(self, idx):
        return self.camera_params_opencv[idx], self.images[idx], self.silhouettes[idx], self.manos[idx]


if __name__ == '__main__':
    data_path = '/home/azhuavlev/Desktop/Data/InterHand_Neuman/03'

    all_ids = list(range(len(
        glob.glob(os.path.join(data_path, 'images', '*.png'))
    )))

    # use 80% of the data for training, randomize the order
    np.random.shuffle(all_ids)
    train_ids = all_ids[:int(0.7 * len(all_ids))]
    test_ids = all_ids[int(0.7 * len(all_ids)):]
    # train_ids = all_ids[:10]
    # test_ids = all_ids[10:]
    print(test_ids)

    # We sample 1 random camera in a minibatch.
    batch_size = 1

    # # Use dataset of single image for debugging
    # full_dataset = dataset_single_image.NeumanDataset(data_path, all_ids)
    # train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    # test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=5)
    # full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=5)

    # double_all_ids = all_ids + all_ids
    # print(double_all_ids)
    # exit()
    train_dataset = NeumanDataset(data_path, train_ids)
    test_dataset = NeumanDataset(data_path, test_ids)
    full_dataset = NeumanDataset(data_path,
                                                    all_ids
                                                    )

    import matplotlib.pyplot as plt
    for i in range(10):
        camera_params, images, silhouettes, manos = full_dataset[i]
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(images)
        ax[1].imshow(silhouettes)
        plt.show()
        plt.close()