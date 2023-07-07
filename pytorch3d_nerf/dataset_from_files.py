import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2
import mano_pytorch3d

class NeumanDataset(torch.utils.data.Dataset):
    def __init__(self, exp_dir, cap_ids):
        self.exp_dir = exp_dir
        self.cap_ids = cap_ids

        self.load_cameras()
        self.load_images()
        self.load_silhouettes()
        self.load_mano()


    def load_cameras(self):
        cam_path = self.exp_dir + '/cameras'

        self.camera_params_opencv = []
        for i in self.cap_ids:
            json_file = cam_path + f'/{i:05d}.json'
            with open(json_file) as f:
                data = json.load(f)

                # print(data)

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
                image_size = np.array([512, 334], dtype=np.float32)

                self.camera_params_opencv.append({
                    'R': R,
                    't': t,
                    'intrinsic_mat': intrinsic_mat,
                    'image_size': image_size
                })

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

            self.silhouettes.append(sil)

    def load_mano(self):
        mano_path = self.exp_dir + '/mano'

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
            mano_pose = torch.FloatTensor(mano['pose']).view(-1, 3)

            # root pose is at position 0, pose of rest of the hand is at positions [1:]
            root_pose = mano_pose[0]
            hand_pose = mano_pose[1:, :].view(-1)

            # get betas (called shapes here) and translation vector
            shape = torch.FloatTensor(mano['shape'])
            trans = torch.FloatTensor(mano['trans'])

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

            mano_dict = {
                'root_pose': root_pose,
                'hand_pose': hand_pose,
                'shape': shape,
                'trans':trans,
                'verts': vertices_py3d.squeeze(0),
                'Ts': Ts_xyz.squeeze(0),
            }
            # print(mano_dict)
            self.manos.append(mano_dict)

    def __len__(self):
        return len(self.camera_params_opencv)

    def __getitem__(self, idx):
        return self.camera_params_opencv[idx], self.images[idx], self.silhouettes[idx], self.manos[idx]