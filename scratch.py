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
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

