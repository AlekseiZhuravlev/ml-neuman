import torch
import torch.nn as nn
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
)

def get_look_at_view_R_T(n_cameras, random_cameras, device):
    # get trajectory rotating around 0,0,0

    Rs = torch.tensor([], device=device)
    Ts = torch.tensor([], device=device)

    for i in range(n_cameras):
        if random_cameras:
            R, T = look_at_view_transform(
                dist=0.3,
                elev = 360.0 * torch.rand(1, device=device)[0],
                azim=360.0 * torch.rand(1, device=device)[0],
                device=device
            )
        else:
            R, T = look_at_view_transform(
                dist=0.3,
                elev=360.0 / n_cameras * i,
                azim=360.0 / n_cameras * i,
                device=device
            )
        Rs = torch.cat([Rs, R], dim=0)
        Ts = torch.cat([Ts, T], dim=0)
    return Rs, Ts

def create_canonical_cameras(n_cameras, random_cameras, device):
    Rs, Ts = get_look_at_view_R_T(n_cameras, random_cameras, device)
    # image_sizes = torch.tensor([[512.0, 334.0]], device=device).repeat(n_cameras, 1)

    batch_cameras = FoVPerspectiveCameras(
        R=Rs,
        T=Ts,
        znear=0.01,
        zfar=10,
        device=device,
        # image_size=image_sizes
    )
    return batch_cameras

if __name__ == '__main__':
    batch_cameras = create_canonical_cameras(10, random_cameras=False, device='cpu')
    print(batch_cameras.get_camera_center())

