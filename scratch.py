import joblib
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
# read this file with joblib /home/azhuavlev/Desktop/Data/neuman/data/seattle/smpl_output_romp.pkl

# data = joblib.load('/home/azhuavlev/Desktop/Data/neuman/data/seattle/smpl_output_romp.pkl')
# print(data[1].keys())
# for k in data[1].keys():
#     print(k)
#     print(data[1][k].shape)
# print(data[1]['betas'].shape)
#
# def extract_smpl_at_frame(raw_smpl, frame_id):
#     out = {}
#     for k, v in raw_smpl.items():
#         try:
#             out[k] = v[frame_id]
#         except:
#             out[k] = None
#     return out
#
# # read file /home/azhuavlev/Desktop/Data/neuman/data/seattle/alignments.npy
# data = np.load('/home/azhuavlev/Desktop/Data/neuman/data/seattle/alignments.npy', allow_pickle=True)
# print(data.item()['00010.png'])

# read /home/azhuavlev/Desktop/Data/Interhand_masked/annotations/test/InterHand2.6M_test_MANO_NeuralAnnot.json
with open('/home/azhuavlev/Desktop/Data/Interhand_masked/annotations/test/InterHand2.6M_test_MANO_NeuralAnnot.json', 'r') as f:
    mano_data_full = json.load(f)

with open('/home/azhuavlev/Desktop/Data/Interhand_masked/annotations/test/InterHand2.6M_test_joint_3d.json', 'r') as f:
    joint_data_full = json.load(f)
# print(mano_data['0']['18655'])
mano_param = mano_data_full['0']['18655']['left']

mano_pose = torch.FloatTensor(mano_param['pose']).view(-1, 3)
root_pose = mano_pose[0].view(1, 3)
hand_pose = mano_pose[1:, :].view(1, -1)

trans = torch.FloatTensor(mano_param['trans']).view(1,3)

print('root_pose', root_pose)
print('hand_pose', hand_pose)
print('trans', trans)

# plot mano_pose as 3d plot, mark each point with a number
# make 3d plot
fig = plt.figure()
axs = fig.add_subplot(111, projection='3d')

# axs.scatter(mano_pose[:, 0], mano_pose[:, 1], mano_pose[:, 2])
# for i in range(mano_pose.shape[0]):
#     axs.text(mano_pose[i, 0], mano_pose[i, 1], mano_pose[i, 2], str(i))
# axs.set_xlabel('x')
# axs.set_ylabel('y')
# axs.set_zlabel('z')
#
# plt.show()

# axs.scatter(trans[:, 0], trans[:, 1], trans[:, 2])
# axs.set_xlabel('x')
# axs.set_ylabel('y')
# axs.set_zlabel('z')
#
# plt.show()

# print(joint_data_full['0']['18655']['world_coord'])
joint_coords = np.array(joint_data_full['0']['18655']['world_coord']) / 1000

axs.scatter(joint_coords[:, 0], joint_coords[:, 1], joint_coords[:, 2])
for i in range(joint_coords.shape[0]):
    axs.text(joint_coords[i, 0], joint_coords[i, 1], joint_coords[i, 2], str(i))
axs.set_xlabel('x')
axs.set_ylabel('y')
axs.set_zlabel('z')

plt.show()