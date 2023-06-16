import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import pickle

from torch import inf
# from kaolin.ops.mesh import check_sign


def display_sampled_rays_nerf():
    # read this matplotlib figure and display it '/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/test_rays.fig.pickle'
    with open('/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/input_rays_batch_1685759.fig.pickle', 'rb') as f:
        fig = pickle.load(f)

    with open('/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/output_rays_batch_1685759.fig.pickle', 'rb') as f:
        fig2 = pickle.load(f)

    plt.show()



if __name__ == '__main__':
    # cpt = torch.load('/home/azhuavlev/Desktop/Results/neuman-mano/interhand/lightning_logs/lightning_logs/version_20/checkpoints/epoch=9-step=400.ckpt')
    # cpt = torch.load('/home/azhuavlev/Desktop/Data/neuman/out/seattle_human/checkpoint.pth.tar')
    # print(cpt.keys())
    # print(cpt['hybrid_model_state_dict'].keys())

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # verts = torch.tensor([[[0., 0., 0.],
    #                        [1., 0.5, 1.],
    #                        [0.5, 1., 1.],
    #                        [1., 1., 0.5]]], device=device)
    # faces = torch.tensor([[0, 3, 1],
    #                       [0, 1, 2],
    #                       [0, 2, 3],
    #                       [3, 2, 1]], device=device)
    # axis = torch.linspace(0.1, 0.9, 3, device=device)
    # p_x, p_y, p_z = torch.meshgrid(axis + 0.01, axis + 0.02, axis + 0.03)
    # points = torch.cat((p_x.unsqueeze(-1), p_y.unsqueeze(-1), p_z.unsqueeze(-1)), dim=3)
    # points = points.view(1, -1, 3)
    # print(check_sign(verts, faces, points))
    #
    # exit()
    #
    #
    # with open('/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/inside_volume.pkl', 'rb') as f:
    #     fig2 = pickle.load(f)

    with open('/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/mesh.pkl', 'rb') as f:
        fig2 = pickle.load(f)

    plt.show()

