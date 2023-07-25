import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
    look_at_view_transform
)
from pytorch3d.implicitron.dataset.rendered_mesh_dataset_map_provider import RenderedMeshDatasetMapProvider

class CanSpaceDataset(torch.utils.data.Dataset):
    def __init__(self, n_cameras):
        self.n_cameras = n_cameras
        self.Rs, self.Ts = None, None

        self.generate_cameras()

    def generate_cameras(self):
        # logRs = torch.zeros(self.n_cameras, 3)
        # logRs[:, 1] = torch.linspace(-3.14, 3.14, self.n_cameras)
        # self.Rs = so3_exp_map(logRs)
        #
        # self.Ts = torch.zeros(self.n_cameras, 3)
        # self.Ts[:, 2] = 0.3

        cow_provider = RenderedMeshDatasetMapProvider(
            data_file="data/cow_mesh/cow.obj",
            use_point_light=False,
            resolution=80,
        )
        dataset_map = cow_provider.get_dataset_map()
        tr_cameras = [training_frame.camera for training_frame in dataset_map.train]

        self.Rs = torch.stack([tr_camera.R[0] for tr_camera in tr_cameras])
        self.Ts = torch.stack([tr_camera.T[0] for tr_camera in tr_cameras])
        self.Ts[:, 2] = 0.3
        print(self.Ts)


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