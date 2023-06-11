import joblib
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import tensorboardX
import pickle


def display_sampled_rays_nerf():
    # read this matplotlib figure and display it '/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/test_rays.fig.pickle'
    with open('/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/input_rays_batch_1685759.fig.pickle', 'rb') as f:
        fig = pickle.load(f)

    with open('/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/output_rays_batch_1685759.fig.pickle', 'rb') as f:
        fig2 = pickle.load(f)

    plt.show()



if __name__ == '__main__':
    cpt = torch.load('/home/azhuavlev/Desktop/Results/neuman-mano/interhand/lightning_logs/lightning_logs/version_20/checkpoints/epoch=9-step=400.ckpt')
    # cpt = torch.load('/home/azhuavlev/Desktop/Data/neuman/out/seattle_human/checkpoint.pth.tar')
    print(cpt.keys())
    # print(cpt['hybrid_model_state_dict'].keys())

