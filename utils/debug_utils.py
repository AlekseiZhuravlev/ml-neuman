# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/utils/debug_utils.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE
from utils import ray_utils
import torch
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def embed_breakpoint(debug_info='', terminate=True):
    print('\nyou are inside a break point')
    if debug_info:
        print(f'debug info: {debug_info}')
    print('')
    embedding = ('import numpy as np\n'
                 'import IPython\n'
                 'import matplotlib.pyplot as plt\n'
                 'IPython.embed()\n'
                 )
    if terminate:
        embedding += (
            'exit()'
        )

    return embedding


def draw_pointcloud_plot(batch):
    origins = batch['direction'].cpu().detach().numpy()
    colors = fine_rgb_map.cpu().detach().numpy()
    # make a 3d plot of the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c=colors)

    with open('/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/output_rays_batch.fig.pickle', 'wb') as f:
        pickle.dump(fig, f)

    colors = batch['color'].cpu().detach().numpy()
    # make a 3d plot of the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c=colors)

    with open('/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/images/input_rays_batch.fig.pickle', 'wb') as f:
        pickle.dump(fig, f)


def plot_warped_and_orig(human_pts, mesh, raw_Ts, Ts, name):
    human_pts_old = human_pts.clone()

    can_pts = (Ts @ ray_utils.to_homogeneous(human_pts)[..., None])[:, :3, 0]  # .reshape(human_b, human_n,
    #                                                                            #       3)

    # plot mesh as 3d point cloud
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()

    human_pts = human_pts.cpu().detach().numpy()[:1000, :]
    mesh = mesh[0].cpu().detach().numpy()
    raw_Ts = torch.inverse(raw_Ts[0]).cpu().detach().numpy()

    can_mesh = (raw_Ts @ ray_utils.to_homogeneous(mesh)[..., None])[:, :3, 0]

    # set alpha to 0.1 for human points

    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(human_pts[:, 0], human_pts[:, 1], human_pts[:, 2], c='r', marker='o', alpha=1, s=0.5)

    # ax = fig.add_subplot(112, projection='3d')
    ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], c='b', marker='o', s=0.5)

    ax.set_title('human_pts (red) and mesh (blue) in observation space')

    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(can_mesh[:, 0], can_mesh[:, 1], can_mesh[:, 2], c='g', marker='o', s=0.5)

    can_pts = can_pts.cpu().detach().numpy().reshape(-1, 3)[:1000, :]
    print('can_pts.shape', can_pts.shape, can_pts)
    # exit()
    ax.scatter(can_pts[:, 0], can_pts[:, 1], can_pts[:, 2], c='y', marker='o', alpha=1, s=0.5)

    ax.set_title('can_mesh (green) and can_pts (yellow) in canonical space')

    print('human_pts.shape', human_pts.shape, human_pts)
    print('mesh.shape', mesh.shape, mesh)

    # print('human_near', batch['human_near'])
    # print('human_far', batch['human_far'])
    # print('cap_id', batch['cap_id'])

    # save figure as pickle
    import pickle
    with open(f'/home/azhuavlev/PycharmProjects/ml-neuman_mano/out/{name}.pkl', 'wb') as f:
        pickle.dump(fig, f)