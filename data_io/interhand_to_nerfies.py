import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

import sys

sys.path.append('../..')


class InterhandToNerfiesConverter:
    def __init__(self, basefolder, split, capture_n, pose,
                 frames_list_slice,
                 cameras_list, val_cameras_frac, max_cameras, experiment_n):
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
        if cameras_list:
            self.cameras_list = cameras_list
        else:
            self.cameras_list = [folder_name[3:] for folder_name in sorted(os.listdir(self.pose_path))]

        self.frames_list_slice = frames_list_slice

        # split cameras into train and val,
        self.train_cameras, self.val_cameras = self.split_cameras(val_cameras_frac, max_cameras)
        print(f'Using {len(self.train_cameras)} cameras for training and {len(self.val_cameras)} cameras for validation')

        self.train_img_ids = []
        self.val_img_ids = []

        self.target_folder = '/itet-stor/azhuavlev/net_scratch/Projects/Data/InterHand_Nerfies_format/' \
                             f'{experiment_n}'
        self.camera_path = self.target_folder + '/camera'
        self.rgb_path = self.target_folder + '/rgb/4x'

        # check that all cameras have the same number of images
        self.clear_root_folder()

        os.makedirs(self.target_folder, exist_ok=True)
        os.makedirs(self.camera_path, exist_ok=True)
        os.makedirs(self.rgb_path, exist_ok=True)

        self.metadata = {}  # dict to store time_id, warp_id, appearance_id, camera_id
        self.curr_img = 0

        # load camera parameters
        # with open(self.base_folder + '/human_annot/' + \
        #           f'InterHand2.6M_{self.split}_camera.json', 'r') as f:
        #     self.camera_params_dict = json.load(f)

        with open(self.base_folder + '/annotations/' + self.split + '/InterHand2.6M_' + split + '_camera.json') as f:
            self.camera_params_dict = json.load(f)

    def check_camera_img_count(self):
        """
        Check that all cameras have the same number of images
        """
        camera_img_count = []
        for camera in self.cameras_list:
            camera_folder = self.pose_path + '/' + f'cam{camera}'
            camera_img_count.append(len(os.listdir(camera_folder)))

        # if len(set(camera_img_count)) != 1:
        #     for i, camera in enumerate(self.cameras_list):
        #         print(f'Camera {camera} has {camera_img_count[i]} images')
        #     raise ValueError('Cameras have different number of images')

        # remove cameras from self.cameras_list if they have less images than the majority of cameras
        max_img_count = max(camera_img_count)
        for i, camera in enumerate(self.cameras_list):
            if camera_img_count[i] < max_img_count:
                self.cameras_list.remove(camera)
                print(f'Removing camera {camera} from the list of cameras to use, it has {camera_img_count[i]} images'
                      f' while the majority of cameras have {max_img_count} images')

        print('All cameras have the same number of images')
        return camera_img_count[0]

    def split_cameras(self, val_cameras_frac, max_cameras):
        """
        Split cameras into train and val randomly
        :param val_cameras_frac: fraction of cameras to use for validation
        :param max_cameras: maximum number of cameras to use
        """
        if not max_cameras or max_cameras > len(self.cameras_list):
            max_cameras = len(self.cameras_list)

        n_val_cameras = int(max_cameras * val_cameras_frac)
        val_cameras = set(np.random.choice(self.cameras_list, n_val_cameras, replace=False))

        # make a set of remaining cameras and choose max_cameras - n_val_cameras cameras from it as train cameras
        train_cameras = set(self.cameras_list) - val_cameras
        train_cameras = set(np.random.choice(list(train_cameras), max_cameras - n_val_cameras, replace=False))

        # train_cameras = set()
        # for camera in self.cameras_list:
        #     if camera not in val_cameras:
        #         train_cameras.add(camera)
        #     if len(train_cameras) >= max_cameras - n_val_cameras:
        #         break

        return train_cameras, val_cameras


    def copy_images(self):
        """
        Copy images from interhand to nerfies format
        """

        n_images = self.check_camera_img_count()

        for j_image in tqdm(range(n_images)[self.frames_list_slice]):
            # iterate over self.cameras_list in normal order on odd images and in reverse order on even images
            iteration_factor = 1 if j_image % 2 == 0 else -1

            for i_camera, camera in enumerate(self.cameras_list[::iteration_factor]):
                if camera in self.train_cameras:
                    self.train_img_ids.append(self.curr_img)
                elif camera in self.val_cameras:
                    self.val_img_ids.append(self.curr_img)
                else:
                    continue

                camera_folder = self.pose_path + '/' + f'cam{camera}'
                img = sorted(os.listdir(camera_folder))[j_image]
                self.copy_image(img, camera_folder)

                self.create_camera_file(camera)
                self.update_metadata_unique(i_camera, j_image)
                # self.update_metadata(i_camera, j_image)

                self.curr_img += 1


        # copy images to rgb folder and create camera files
        # for i_camera, camera in enumerate(self.cameras_list):
        #
        #     camera_folder = self.pose_path + '/' + f'cam{camera}'
        #     for j_image, img in enumerate(tqdm(sorted(os.listdir(camera_folder)))):
        #         # copy image
        #         self.copy_image(img, camera_folder)
        #
        #         self.create_camera_file(camera)
        #
        #         self.update_metadata(i_camera, j_image)
        #
        #         self.curr_img += 1

        # save metadata to file
        with open(f'{self.target_folder}/metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=4)

        # create test cameras

        # create dataset file
        self.create_dataset_file()

        # create scene file
        self.create_scene_file()

    def clear_root_folder(self):
        """
        Clear root folder from previous experiments
        """
        os.system(f'rm -rf {self.target_folder}/*')


    def update_metadata(self, i_camera, j_image):
        # update metadata
        self.metadata[f'{self.curr_img:05d}'] = {
            "time_id": j_image,
            "warp_id": j_image,
            "appearance_id": j_image,
            "camera_id": i_camera,
        }

    def update_metadata_unique(self, i_camera, j_image):
        # update metadata
        self.metadata[f'{self.curr_img:05d}'] = {
            "time_id": self.curr_img,
            "warp_id": self.curr_img,
            "appearance_id": self.curr_img,
            "camera_id": self.curr_img,
        }

    def copy_image(self, img, camera_folder):
        # copy image to rgb folder
        os.system(f'cp {camera_folder}/{img} {self.rgb_path}/{img}')

        # rename copied image to curr_img, padded to 4 digits
        os.system(f'mv {self.rgb_path}/{img} {self.rgb_path}/{self.curr_img:05d}.jpg')

    def create_camera_file(self, camera):

        # load camera parameters
        campos = self.camera_params_dict[self.capture_n]['campos'][camera]
        camrot = self.camera_params_dict[self.capture_n]['camrot'][camera]
        focal = self.camera_params_dict[self.capture_n]['focal'][camera]
        princpt = self.camera_params_dict[self.capture_n]['princpt'][camera]

        # open image
        img_jpg = Image.open(f'{self.rgb_path}/{self.curr_img:05d}.jpg')

        # img_jpg.show()
        # img_jpg.save(f'example.png')
        # exit(
        img = img_jpg

        # pad image with 50 black pixel on the left
        # img = Image.new('RGB', (img_jpg.size[0] + 50, img_jpg.size[1]), (0, 0, 0))
        # img.paste(img_jpg, (50, 0))

        # invert image colors
        # img_arr = np.array(img)
        # img_arr = 255 - img_arr
        # img = Image.fromarray(img_arr)

        # save image as png and remove jpg
        img.save(f'{self.rgb_path}/{self.curr_img:05d}.png')
        os.system(f'rm {self.rgb_path}/{self.curr_img:05d}.jpg')

        # get image size
        width, height = img.size

        # create camera parameters dict
        campos = np.array(campos)
        camera_params = {
            'orientation': camrot,
            'position': list(campos / (np.linalg.norm(campos) + 0.000001)),
            'focal_length': focal[0],
            'principal_point': princpt,
            'image_size': [width * 4, height * 4],
            'skew': 0,
            'pixel_aspect_ratio': 1,
            'radial_distortion': [1e-4, 1e-4, 1e-4],
            'tangential_distortion': [1e-4, 1e-4]
        }
        # save camera parameters to file
        with open(f'{self.camera_path}/{self.curr_img:05d}.json', 'w') as f:
            json.dump(camera_params, f, indent=4)

    def create_dataset_file(self):
        dataset_params = {
            "count": self.curr_img,
            "num_exemplars": self.curr_img,
            "ids": [f"{i:05d}" for i in range(self.curr_img)],
            "train_ids": [f"{i:05d}" for i in self.train_img_ids],
            "val_ids": [f"{i:05d}" for i in self.val_img_ids]
        }
        with open(f'{self.target_folder}/dataset.json', 'w') as f:
            json.dump(dataset_params, f, indent=4)

    def create_scene_file(self):
        scene_params = {
          "scale": 1,
          "scene_to_metric": 1,
          "center": [0, 0, 0],
          "near": 0.1,
          "far": 500
        }
        with open(f'{self.target_folder}/scene.json', 'w') as f:
            json.dump(scene_params, f, indent=4)


if __name__ == '__main__':
    converter = InterhandToNerfiesConverter(
        basefolder='/home/azhuavlev/Desktop/Projects/Data/Interhand_masked',
        split='test',
        capture_n='0',
        pose='ROM04_LT_Occlusion',
        frames_list_slice=slice(None, 100, None),
        cameras_list=None,#['400262', '400263', '400264', '400265', '400284'],
        val_cameras_frac=0.1,
        max_cameras=15,
        experiment_n='09_15_cameras_unique_warp'
    )
    converter.copy_images()

    exit(0)
    # make video
    print(pathlib.Path(converter.rgb_path))
    imgs = make_video.load_all_images_from_dir(pathlib.Path(converter.rgb_path), 'png')
    print(len(imgs))
    concatenated_imgs = make_video.concatenate_images(imgs[:len(imgs) // 2], imgs[len(imgs) // 2:])
    print(len(concatenated_imgs))
    make_video.save_video(pathlib.Path(converter.target_folder), concatenated_imgs, fps=5)