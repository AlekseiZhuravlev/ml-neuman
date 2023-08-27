import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

import sys

import numpy as np
# sys.path.append('../..')


def clear_folder(folder):
    """
    Clear root folder from previous experiments
    """
    os.system(f'rm -rf {folder}/*')


class InterhandToNeumanConverter:
    def __init__(self, basefolder, split, capture_n, pose,
                 cameras_list, experiment_n, max_images_per_camera, max_cameras, every_n_frames):
        """
        :param basefolder: path to the InterHand dataset
        :param split: 'train', 'test' or 'val'
        :param capture_n: capture number
        :frames_list_slice: during iteration over the frames for each camera, only use frames from this slice
        :param pose: hand pose
        :param cameras_list: list of cameras to use. If None, all cameras are used
        :param val_cameras_frac: fraction of cameras to use for validation
        :param max_cameras: maximum number of cameras to use if cameras_list is None
        :param experiment_n: experiment number
        """
        self.split = split
        self.capture_n = capture_n
        self.pose = pose

        self.base_folder = basefolder
        self.pose_path = self.base_folder + '/' + 'images' + '/' + split + '/' + \
                         f"Capture{capture_n}" + '/' + pose
        self.max_cameras = max_cameras
        self.every_n_frames = every_n_frames

        if cameras_list:
            self.cameras_list = cameras_list
        else:
            self.cameras_list = [folder_name[3:] for folder_name in sorted(os.listdir(self.pose_path))]

        cameras_to_exclude = ['400267', '400268']
        for camera_to_exclude in cameras_to_exclude:
            if camera_to_exclude in self.cameras_list:
                self.cameras_list.remove(camera_to_exclude)

        self.cameras_list = self.cameras_list[:self.max_cameras]
        print('cameras_list', self.cameras_list)

        self.target_folder = '/itet-stor/azhuavlev/net_scratch/Projects/Data/InterHand_Neuman/' \
                             f'{experiment_n}'

        self.camera_path = self.target_folder + '/cameras'
        self.rgb_path = self.target_folder + '/images'
        self.mano_path = self.target_folder + '/mano'
        self.joints_path = self.target_folder + '/joints'

        # check that all cameras have the same number of images
        clear_folder(self.target_folder)

        os.makedirs(self.target_folder, exist_ok=True)
        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.mano_path, exist_ok=True)
        os.makedirs(self.camera_path, exist_ok=True)
        os.makedirs(self.joints_path, exist_ok=True)

# '/home/azhuavlev/Desktop/Data/Interhand_masked/annotations/test/'
        with open(self.base_folder + f'/annotations/{self.split}/InterHand2.6M_{self.split}_MANO_NeuralAnnot.json',
                  'r') as f:
            self.mano_dict = json.load(f)
        with open(self.base_folder + f'/annotations/{self.split}/InterHand2.6M_{self.split}_joint_3d.json',
                  'r') as f:
            self.joint_dict = json.load(f)

        self.max_images_per_camera = max_images_per_camera

        with open(self.base_folder + '/annotations/' + self.split + '/InterHand2.6M_' + split + '_camera.json') as f:
            self.camera_params_dict = json.load(f)

        self.curr_img = 0

    def check_camera_img_count(self):
        """
        Check that all cameras have the same number of images
        """
        camera_img_count = []
        for camera in self.cameras_list:
            camera_folder = self.pose_path + '/' + f'cam{camera}'
            camera_img_count.append(len(os.listdir(camera_folder)))

        # remove cameras from self.cameras_list if they have less images than the majority of cameras
        max_img_count = max(camera_img_count)
        for i, camera in enumerate(self.cameras_list):
            if camera_img_count[i] < max_img_count:
                self.cameras_list.remove(camera)
                print(f'Removing camera {camera} from the list of cameras to use, it has {camera_img_count[i]} images'
                      f' while the majority of cameras have {max_img_count} images')

        print('All cameras have the same number of images')
        return camera_img_count[0]

    def copy_images(self):
        """
        Copy images from interhand to neuman format
        """
        # n_images_per_camera = self.check_camera_img_count()
        # n_images = min(n_images_per_camera, self.max_images_per_camera)

        tqdm_iterable = tqdm(self.cameras_list)
        for i_camera, camera in enumerate(tqdm_iterable):

            # how many images does this camera have
            n_imgs_camera_has = len(os.listdir(self.pose_path + '/' + f'cam{camera}'))

            # how many images to copy from this camera
            if n_imgs_camera_has < self.max_images_per_camera * self.every_n_frames:
                n_images = n_imgs_camera_has // self.every_n_frames
            else:
                n_images = self.max_images_per_camera

            for j_image_in_folder in range(0, n_images * self.every_n_frames, self.every_n_frames):

                if n_images == 1:
                    print(f'n_images == 1, setting j_image_in_folder to {self.every_n_frames}')
                    j_image_in_folder = self.every_n_frames

                # print('camera', camera, 'j_image_in_folder', j_image_in_folder)
                tqdm_iterable.set_description(f'camera {camera}, j_image_in_folder {j_image_in_folder}')
                try:
                    camera_folder = self.base_folder + '/images/' + self.split + '/' + \
                                    f"Capture{self.capture_n}" + '/' + self.pose + '/' + f'cam{camera}'
                    img = sorted(os.listdir(camera_folder))[j_image_in_folder]

                    self.copy_image(
                        from_path=camera_folder,
                        img_name=img,
                        to_path=self.target_folder + '/images',
                        grayscale=False
                    )
                    self.create_mano(img)

                    self.create_camera(camera)
                    self.curr_img += 1
                except Exception as e:
                    print(f'Skipping image {img} from camera {camera}, reason: {e}')


    def copy_image(self, from_path, img_name, to_path, grayscale):
        os.makedirs(to_path, exist_ok=True)
        # copy image to rgb folder
        os.system(f'cp {from_path}/{img_name} {to_path}/{img_name}')

        img_jpg = Image.open(f'{to_path}/{img_name}')

        # check if image is > 95% black
        img_np = np.array(img_jpg)
        if np.sum(img_np) < 0.05 * img_np.shape[0] * img_np.shape[1] * img_np.shape[2]:
            raise RuntimeError(f'Image {img_name} is too dark')

        if grayscale:
            img_jpg = img_jpg.convert('L')

        # save image as png and remove jpg
        img_jpg.save(f'{to_path}/{self.curr_img:05d}.png')
        os.system(f'rm {to_path}/{img_name}')


    def create_mano(self, img_name):
        img_id = img_name[5:-4]
        mano_params = self.mano_dict[self.capture_n][img_id]

        # add key 'pose_id' to mano_params
        # mano_params['left']['pose_id'] = k_image_local_idx

        with open(f'{self.mano_path}/{self.curr_img:05d}.json', 'w') as f:
            json.dump(mano_params, f)

        joint_params = self.joint_dict[self.capture_n][img_id]
        with open(f'{self.joints_path}/{self.curr_img:05d}.json', 'w') as f:
            json.dump(joint_params, f)

    def create_camera(self, camera):
        # load camera parameters
        campos = self.camera_params_dict[self.capture_n]['campos'][camera]
        camrot = self.camera_params_dict[self.capture_n]['camrot'][camera]
        focal = self.camera_params_dict[self.capture_n]['focal'][camera]
        princpt = self.camera_params_dict[self.capture_n]['princpt'][camera]

        # create camera parameters dict
        camera_params = {
            'camrot': camrot,
            'campos': campos,
            'focal': focal,
            'princpt': princpt,
        }

        # save camera parameters to file
        with open(f'{self.camera_path}/{self.curr_img:05d}.json', 'w') as f:
            json.dump(camera_params, f, indent=4)


if __name__ == '__main__':
    converter = InterhandToNeumanConverter(
        basefolder='/home/azhuavlev/Desktop/Data/InterHand',
        split='test',
        capture_n='0',
        pose='ROM03_LT_No_Occlusion',
        cameras_list=None,#['400262', '400263', '400264', '400265', '400284'],
        experiment_n='10_images50_cameras15_every5---ROM03_LT_No_Occlusion',
        max_images_per_camera=50,
        max_cameras=15,
        every_n_frames=5,

    )
    converter.copy_images()