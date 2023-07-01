import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeumanDataset(torch.utils.data.Dataset):
    def __init__(self, scene, cap_ids):
        self.scene = scene


        self.Rs = []
        self.Ts = []
        self.images, self.silhouettes = [], []
        for i in cap_ids:
            self.Rs.append(self.scene.captures[i].cam_pose.rotation_matrix[0:3, 0:3])
            self.Ts.append(self.scene.captures[i].cam_pose.translation_vector)

            # not normalized!
            self.images.append(self.scene.captures[i].image)
            self.silhouettes.append(self.scene.captures[i].mask)

        self.Rs = torch.tensor(np.array(self.Rs))
        self.Ts = torch.tensor(np.array(self.Ts))
        self.images = torch.tensor(np.array(self.images))
        self.silhouettes = torch.tensor(np.array(self.silhouettes))

        self.images = (self.images.to(torch.float32) / 255.0).clamp(0.0, 1.0)
        self.silhouettes = self.silhouettes.to(torch.float32)

        print('R', self.Rs.shape)
        print('T', self.Ts.shape)
        print('images', self.images.shape, self.images.dtype)
        print('silhouettes', self.silhouettes.shape, self.silhouettes.dtype)

    def __len__(self):
        return len(self.Rs)

    def __getitem__(self, idx):
        return self.Rs[idx], self.Ts[idx], self.images[idx], self.silhouettes[idx]