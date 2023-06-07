# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/utils/debug_utils.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


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