import sys, os
import glob
sys.path.append("/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf")

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2
from mano_custom import mano_pytorch3d
from scipy import ndimage
from pytorch3d.transforms.so3 import so3_exponential_map, so3_log_map
from losses.canonical_utils.cameras_canonical import get_look_at_view_R_T
from losses.canonical_utils.render_canonical import RendererCanonical
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
)
import segmentation.grabcut_utils as grabcut_utils
import shutil


class NeumanDataset(torch.utils.data.Dataset):
    def __init__(self, exp_dir, cap_ids, bg_rm_dilation, use_grabcut):
        self.exp_dir = exp_dir
        self.cap_ids = cap_ids
        self.hand_model = mano_pytorch3d.create_mano_custom(return_right_hand=False)

        print('loading cameras')
        self.load_cameras_interhand()
        # self.load_silhouettes()
        print('loading images')
        self.load_images()
        print('loading mano')
        self.load_mano()

        print('creating zero pose silhouettes')
        self.create_zero_pose_silhouettes()
        print('creating silhouettes')
        self.create_silhouettes()

        self.use_grabcut = use_grabcut
        if use_grabcut:
            print('applying grabcut to silhouettes')
            self.load_grabcut_silhouettes()
            self.silhouettes = self.silhouettes_grabcut
        else:
            print('not applying grabcut to silhouettes')
            self.silhouettes = self.silhouettes_raw

        print('masking images')
        self.mask_images(bg_rm_dilation)



    def load_cameras_interhand(self):
        cam_path = self.exp_dir + '/cameras'

        self.camera_params_interhand = []
        self.camera_params_training = []

        self.Rs_can, self.Ts_can = get_look_at_view_R_T(len(self.cap_ids), random_cameras=False, device='cpu')

        for j_index, i in enumerate(self.cap_ids):
            json_file = cam_path + f'/{i:05d}.json'
            with open(json_file) as f:
                data = json.load(f)

                campos = np.array(data['campos'], dtype=np.float32).reshape(3) / 1000.0
                camrot = np.array(data['camrot'], dtype=np.float32).reshape(3, 3)
                focal = np.array(data['focal'], dtype=np.float32).reshape(2)
                princpt = np.array(data['princpt'], dtype=np.float32).reshape(2)

                R = camrot
                t = -np.dot(camrot, campos.reshape(3, 1)).reshape(3)  # -Rt -> t

                intrinsic_mat = np.array([
                    [focal[0], 0, princpt[0]],
                    [0, focal[1], princpt[1]],
                    [0, 0, 1]
                ], dtype=np.float32)

                # (height, width)
                image_size = np.array([512.0, 334.0], dtype=np.float32)

                self.camera_params_interhand.append({
                    'R': R,
                    't': t,
                    'intrinsic_mat': intrinsic_mat,
                    'image_size': image_size,
                    'focal': focal,
                    'princpt': princpt,
                    'campos': campos,
                    'camrot': camrot,
                })

                R_train = torch.eye(3)
                t_train = torch.zeros(3)
                self.camera_params_training.append({
                    'R_pytorch3d': R_train,
                    't_pytorch3d': t_train,
                    'R_can': self.Rs_can[j_index],
                    't_can': self.Ts_can[j_index],
                    'intrinsic_mat': torch.tensor(intrinsic_mat),
                    'image_size': torch.tensor(image_size),
                    'focal': torch.tensor(focal),
                    'princpt': torch.tensor(princpt),
                })
#
    def load_images(self):
        img_path = self.exp_dir + '/images'

        self.images = []

        for i in self.cap_ids:
            img_file = img_path + f'/{i:05d}.png'
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

            self.images.append(img)

    def load_silhouettes(self):
        raise NotImplementedError('create silhouettes instead')
        sil_path = self.exp_dir + '/segmentations'

        self.silhouettes_raw = []

        for i in self.cap_ids:
            sil_file = sil_path + f'/{i:05d}.png'
            sil = cv2.imread(sil_file, cv2.IMREAD_GRAYSCALE)
            sil = sil.astype(np.float32) / 255.0

            # invert silhouettes
            sil = 1.0 - sil
            sil = sil.clip(0.0, 1.0)

            # mask_dilated_5 = ndimage.binary_dilation(mask, iterations=5) - mask.numpy()
            # sil_dilated_10 = ndimage.binary_dilation(sil, iterations=10).astype(np.float32).clip(0.0, 1.0)

            self.silhouettes_raw.append(
                sil
            )

    def load_mano(self):
        mano_path = self.exp_dir + '/mano'
        joints_path = self.exp_dir + '/joints'

        hand_model = self.hand_model

        self.manos = []
        for j_curr_item, i_cap_id in enumerate(self.cap_ids):
            # print('j_curr_item', j_curr_item, 'i_cap_id', i_cap_id)

            mano_file = mano_path + f'/{i_cap_id:05d}.json'
            with open(mano_file) as f:
                mano = json.load(f)['left']

            root_pose, hand_pose, shape, trans = self.apply_extr_to_mano(mano, hand_model, j_curr_item)

            verts_zero_pose_xyz, Ts_xyz = hand_model.verts_transformations_xyz(
                betas=shape,
                global_orient=root_pose,
                hand_pose=hand_pose,
                transl=trans
            )
            verts_zero_pose_py3d = torch.stack(
                (-verts_zero_pose_xyz[:, :, 0],
                 -verts_zero_pose_xyz[:, :, 1],
                 verts_zero_pose_xyz[:, :, 2]),
                2)
            vertices_py3d = hand_model.forward_pytorch3d(
                betas=shape,
                global_orient=root_pose,
                hand_pose=hand_pose,
                transl=trans
            )

            if 'pose_id' in mano:
                pose_id = mano['pose_id']
            else:
                # print('pose_id not found in mano')
                pose_id = 1

            mano_dict = {
                'root_pose': root_pose.squeeze(0),
                'hand_pose': hand_pose.squeeze(0),
                'shape': shape.squeeze(0),
                'trans':trans.squeeze(0),
                'verts': vertices_py3d.squeeze(0),
                'verts_zero': verts_zero_pose_py3d.squeeze(0),
                'Ts': Ts_xyz.squeeze(0),
                'pose_id': pose_id,
            }
            self.manos.append(mano_dict)

    def apply_extr_to_mano(self, mano_param, hand_model, j_curr_item):

        R = self.camera_params_interhand[j_curr_item]['R']
        t = self.camera_params_interhand[j_curr_item]['t']

        mano_pose = torch.tensor(mano_param['pose'], dtype=torch.float32).view(-1, 3)
        root_pose = mano_pose[0].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))  # multiply camera rotation to MANO root pose
        root_pose = torch.from_numpy(root_pose).view(1, 3)
        hand_pose = mano_pose[1:, :].view(1, -1)
        shape = torch.tensor(mano_param['shape'], dtype=torch.float32).view(1, -1)

        with torch.no_grad():
            output = hand_model(global_orient=root_pose, hand_pose=hand_pose,
                                           betas=shape)  # this is rotation-aligned, but not translation-aligned with the camera coordinates
        mesh = output.vertices[0].detach().numpy()

        joint_regressor = np.load(
            '/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/camera_conversion/J_regressor_mano_ih26m.npy')
        joint_from_mesh = np.dot(joint_regressor, mesh)

        # compenstate rotation (translation from origin to root joint was not cancled)
        root_joint_idx = 20
        root_joint = joint_from_mesh[root_joint_idx, None, :]

        # change translation vector
        trans = np.array(mano_param['trans'])
        trans = np.dot(R, trans.reshape(3, 1)).reshape(1, 3) - root_joint + np.dot(R, root_joint.transpose(1,
                                                                                                           0)).transpose(
            1, 0) + t
        trans = torch.tensor(trans, dtype=torch.float32).view(1, 3)

        return root_pose, hand_pose, shape, trans


    def create_grabcut_silhouette(self, i_image):

        # for i in tqdm(range(len(self.images)), desc='applying grabcut to silhouettes'):
        image_np = (self.images[i_image] * 255).astype(np.uint8)
        silhouette_np = (self.silhouettes_raw[i_image].unsqueeze(-1).cpu().numpy() * 255).astype(np.uint8)

        # refine mask with grabcut
        mano_mask_gc = grabcut_utils.grabcut_refine(silhouette_np, image_np)

        # largest component
        mano_mask_gc_lc = grabcut_utils.largest_component(mano_mask_gc)

        # detect forearm region in the mask (if any)
        mano_mask_forearm_removed = grabcut_utils.remove_forearm(silhouette_np, mano_mask_gc_lc)

        # convert to torch tensor and normalize
        mask_improved = torch.tensor(mano_mask_forearm_removed)
        mask_improved = mask_improved / mask_improved.max()
        return mask_improved

        # self.silhouettes_raw[i] = mask_improved

            # # apply mask to image
            # mask_improved_3ch = mask_improved.unsqueeze(-1).repeat(1, 1, 3)
            # image_segmented = self.images[i] * mask_improved_3ch

    def load_grabcut_silhouettes(self):
        sil_path = self.exp_dir + '/silhouettes_grabcut'
        # create folder if it doesn't exist
        os.makedirs(sil_path, exist_ok=True)

        self.silhouettes_grabcut = []

        for i_image, j_cap_id in enumerate(tqdm(self.cap_ids, desc='loading grabcut silhouettes')):
            sil_file = sil_path + f'/{j_cap_id:05d}.png'

            if not os.path.exists(sil_file):
                try:
                    sil = self.create_grabcut_silhouette(i_image)

                    # check if silhouette is > 90% black
                    sil_np = sil.numpy()
                    if np.sum(sil_np) < 0.005 * sil_np.shape[0] * sil_np.shape[1]:
                        raise RuntimeError(f'Silhouette {j_cap_id} is too dark')
                except Exception as e:
                    print(f'Error applying grabcut to image {j_cap_id}: {e}')
                    sil = self.silhouettes_raw[i_image]
                cv2.imwrite(sil_file, sil.numpy() * 255)
            else:
                sil = cv2.imread(sil_file, cv2.IMREAD_GRAYSCALE)
                sil = sil.astype(np.float32) / 255.0
                sil = sil.clip(0.0, 1.0)

            self.silhouettes_grabcut.append(
                sil
            )


    def mask_images(self, bg_rm_dilation):
        for i in range(len(self.images)):
            if bg_rm_dilation > 0:
                sil_dilated = ndimage.binary_dilation(self.silhouettes[i], iterations=bg_rm_dilation).astype(np.float32).clip(0.0, 1.0)
            else:
                sil_dilated = self.silhouettes[i]

            silh_3ch = np.stack([sil_dilated] * 3, axis=2).astype(np.float32)
            self.images[i] = self.images[i] * silh_3ch

    def create_zero_pose_silhouettes(self):
        # create tensor of indices of self.camera_params_training, and split it into chunks of 150
        idx_tensor = torch.tensor(list(range(len(self.Rs_can))))
        idx_chunks = torch.split(idx_tensor, 150)

        # create silhouettes for each chunk
        self.silhouettes_zero_pose = torch.tensor([], device='cpu')
        for idx_chunk in idx_chunks:
            cameras_chunk = FoVPerspectiveCameras(
                R=torch.stack([self.Rs_can[idx] for idx in idx_chunk], dim=0),
                T=torch.stack([self.Ts_can[idx] for idx in idx_chunk], dim=0),
                znear=0.01,
                zfar=10,
                device='cuda',
            )
            renderer = RendererCanonical(cameras_chunk)

            with torch.no_grad():
                silhouettes_chunk = renderer.render_zero_pose_sil(
                    self.manos[0]['verts_zero'].repeat(len(cameras_chunk), 1, 1).to('cuda'),
                    torch.from_numpy(self.hand_model.faces.astype(np.int32))[None, :, :].to('cuda').repeat(len(cameras_chunk), 1, 1),
                )
            silhouettes_chunk = silhouettes_chunk.cpu().squeeze(-1)
            self.silhouettes_zero_pose = torch.cat((self.silhouettes_zero_pose, silhouettes_chunk), dim=0)

    def create_silhouettes(self):

        # create tensor of indices of self.camera_params_training, and split it into chunks of 150
        idx_tensor = torch.tensor(list(range(len(self.camera_params_training))))
        idx_chunks = torch.split(idx_tensor, 150)

        # create silhouettes for each chunk
        self.silhouettes_raw = torch.tensor([], device='cpu')
        for idx_chunk in idx_chunks:
            cameras_chunk = PerspectiveCameras(
                R=torch.stack([self.camera_params_training[idx]['R_pytorch3d'] for idx in idx_chunk], dim=0),
                T=torch.stack([self.camera_params_training[idx]['t_pytorch3d'] for idx in idx_chunk], dim=0),
                focal_length=torch.stack([self.camera_params_training[idx]['focal'] for idx in idx_chunk], dim=0),
                principal_point=torch.stack([self.camera_params_training[idx]['princpt'] for idx in idx_chunk], dim=0),
                in_ndc=False,
                image_size=torch.stack([self.camera_params_training[idx]['image_size'] for idx in idx_chunk], dim=0),
                device='cuda',
            )
            renderer = RendererCanonical(cameras_chunk)

            verts_to_render = torch.stack([self.manos[idx]['verts'] for idx in idx_chunk], dim=0)

            with torch.no_grad():
                silhouettes_chunk = renderer.render_zero_pose_sil(
                    verts_to_render.to('cuda'),
                    torch.from_numpy(self.hand_model.faces.astype(np.int32))[None, :, :].to('cuda').repeat(len(cameras_chunk), 1, 1),
                )
            silhouettes_chunk = silhouettes_chunk.cpu().squeeze(-1)
            self.silhouettes_raw = torch.cat((self.silhouettes_raw, silhouettes_chunk), dim=0)




        # cameras = PerspectiveCameras(
        #     R=torch.stack([camera_params['R_pytorch3d'] for camera_params in self.camera_params_training], dim=0),
        #     T=torch.stack([camera_params['t_pytorch3d'] for camera_params in self.camera_params_training], dim=0),
        #     focal_length=torch.stack([camera_params['focal'] for camera_params in self.camera_params_training], dim=0),
        #     principal_point=torch.stack([camera_params['princpt'] for camera_params in self.camera_params_training], dim=0),
        #     in_ndc=False,
        #     image_size=torch.stack([camera_params['image_size'] for camera_params in self.camera_params_training], dim=0),
        #     device=device
        # )
        # renderer = RendererCanonical(cameras)
        #
        # verts_to_render = torch.stack([mano['verts'] for mano in self.manos], dim=0)
        #
        # with torch.no_grad():
        #     self.silhouettes = renderer.render_zero_pose_sil(
        #         verts_to_render.to(device),
        #         torch.from_numpy(self.hand_model.faces.astype(np.int32))[None, :, :].to(device).repeat(len(cameras), 1, 1),
        #     )
        # self.silhouettes = self.silhouettes.cpu().squeeze(-1)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.camera_params_training[idx], self.images[idx], self.silhouettes[idx], self.silhouettes_zero_pose[idx], self.manos[idx]


if __name__ == '__main__':

    a = torch.rand(1, 548, 337, 3)
    b = torch.rand(1, 48)

    b = b.unsqueeze(1).repeat(1, 548, 337, 1)

    print(torch.cat([a, b], dim=-1).shape)
    exit(0)


    data_path = '/home/azhuavlev/Desktop/Data/InterHand_Neuman/07_cam5_im12'
    # data_path = '/home/azhuavlev/Desktop/Data/InterHand_Neuman/10_images50_cameras15_every5---ROM04_LT_Occlusion'
    save_path = data_path + '/segmented_images_debug'
    shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path, exist_ok=True)

    all_files = glob.glob(os.path.join(data_path, 'images', '*.png'))
    all_ids = sorted([int(os.path.basename(f)[:-4]) for f in all_files])
    # all_ids = list(range(len(
    #     )
    # )))

    # use 80% of the data for training, randomize the order
    # np.random.shuffle(all_ids)
    # train_ids = all_ids[int(0.3 * len(all_ids)):]
    # test_ids = all_ids[:int(0.3 * len(all_ids))]
    # print(test_ids)

    train_dataset = NeumanDataset(data_path, all_ids, bg_rm_dilation=0, use_grabcut=True)
    camera_params, images, silhouettes, silhouettes_zero, manos = train_dataset[0]
    print(len(manos['root_pose']), len(manos['hand_pose']))
    exit(0)

    import matplotlib.pyplot as plt

    for i in tqdm(range(len(train_dataset)), desc='saving images'):
        # plot images[i] and silhouettes[i] on the same plot, and save to /home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/datasets/images

        batch = train_dataset[i]
        images = batch[1]
        silhouettes = batch[2]

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(images)
        ax[1].imshow(silhouettes)
        plt.savefig(f'{save_path}/{i:05d}.png')
        plt.close(fig)
