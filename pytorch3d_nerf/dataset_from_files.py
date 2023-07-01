import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2

class NeumanDataset(torch.utils.data.Dataset):
    def __init__(self, exp_dir, cap_ids):
        self.exp_dir = exp_dir
        self.cap_ids = cap_ids

        self.load_cameras()
        self.load_images()
        self.load_silhouettes()
        self.load_mano()


        # self.Rs = []
        # self.Ts = []
        # self.images, self.silhouettes = [], []
        # for i in cap_ids:
        #     self.Rs.append(self.scene.captures[i].cam_pose.rotation_matrix[0:3, 0:3])
        #     self.Ts.append(self.scene.captures[i].cam_pose.translation_vector)
        #
        #     # not normalized!
        #     self.images.append(self.scene.captures[i].image)
        #     self.silhouettes.append(self.scene.captures[i].mask)
        #
        # self.Rs = torch.tensor(np.array(self.Rs))
        # self.Ts = torch.tensor(np.array(self.Ts))
        # self.images = torch.tensor(np.array(self.images))
        # self.silhouettes = torch.tensor(np.array(self.silhouettes))
        #
        # self.images = (self.images.to(torch.float32) / 255.0).clamp(0.0, 1.0)
        # self.silhouettes = self.silhouettes.to(torch.float32)
        #
        # print('R', self.Rs.shape)
        # print('T', self.Ts.shape)
        # print('images', self.images.shape, self.images.dtype)
        # print('silhouettes', self.silhouettes.shape, self.silhouettes.dtype)

    def load_cameras(self):
        cam_path = self.exp_dir + '/cameras'

        self.campos = []
        self.camrot = []

        self.Rs = []
        self.Ts = []

        for i in self.cap_ids:
            json_file = cam_path + f'/{i:05d}.json'
            with open(json_file) as f:
                data = json.load(f)

                campos = np.array(data['campos'], dtype=np.float32).reshape(3) / 1000.0
                camrot = np.array(data['camrot'], dtype=np.float32).reshape(3, 3)

                self.campos.append(campos)
                self.camrot.append(camrot)

                R = camrot
                t = -np.dot(camrot, campos.reshape(3, 1)).reshape(3)  # -Rt -> t

                self.Rs.append(R)
                self.Ts.append(t)

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

            mano_dict = {
                'root_pose': root_pose,
                'hand_pose': hand_pose,
                'shape': shape,
                'trans':trans
            }
            print(mano_dict)
            self.manos.append(mano_dict)

    def __len__(self):
        return len(self.Rs)

    def __getitem__(self, idx):
        return self.Rs[idx], self.Ts[idx], self.images[idx], self.silhouettes[idx], self.manos[idx]