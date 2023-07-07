import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import so3_exp_map

class CanSpaceDataset(torch.utils.data.Dataset):
    def __init__(self, n_cameras):
        self.n_cameras = n_cameras
        self.Rs, self.Ts = None, None

        self.generate_cameras()

    def generate_cameras(self):
        logRs = torch.zeros(self.n_cameras, 3)
        logRs[:, 1] = torch.linspace(-3.14, 3.14, self.n_cameras)
        self.Rs = so3_exp_map(logRs)

        self.Ts = torch.zeros(self.n_cameras, 3)
        self.Ts[:, 2] = 0.3


    def __len__(self):
        return len(self.Rs)

    def __getitem__(self, idx):
        return {
            'R': self.Rs[idx],
            't': self.Ts[idx],
            'znear': 0.1,
            'zfar': 100.0,
        }, -1, -1,  {
                'root_pose': -1,
                'hand_pose': -1,
                'shape': -1,
                'trans':-1,
                'verts': -1,
                'Ts': -1,
            }