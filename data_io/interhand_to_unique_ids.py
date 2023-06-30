import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

import sys

# sys.path.append('../..')


def clear_folder(folder):
    """
    Clear root folder from previous experiments
    """
    os.system(f'rm -rf {folder}/*')


class InterhandToNeumanConverter:
    def __init__(self, basefolder, split, capture_n, pose,
                 cameras_list, experiment_n, max_images_per_camera, max_cameras):
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
        self.masked_path = self.base_folder + '/' + 'masked' + '/' + split + '/' + \
                         f"Capture{capture_n}" + '/' + pose
        self.masks_path = self.base_folder + '/' + 'masks' + '/' + split + '/' + \
                         f"Capture{capture_n}" + '/' + pose
        self.depths_path = self.base_folder + '/' + 'depths' + '/' + split + '/' + \
                            f"Capture{capture_n}" + '/' + pose

        self.max_cameras = max_cameras

        if cameras_list:
            self.cameras_list = cameras_list
        else:
            self.cameras_list = [folder_name[3:] for folder_name in sorted(os.listdir(self.pose_path))]
        self.cameras_list = self.cameras_list[:self.max_cameras]

        self.target_folder = '/itet-stor/azhuavlev/net_scratch/Projects/Data/InterHand_Neuman/' \
                             f'{experiment_n}'
        self.camera_path = self.target_folder + '/camera'
        self.rgb_path = self.target_folder + '/images'
        self.mano_path = self.target_folder + '/mano'

        # check that all cameras have the same number of images
        clear_folder(self.target_folder)

        os.makedirs(self.target_folder, exist_ok=True)
        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.mano_path, exist_ok=True)
        os.makedirs(self.camera_path, exist_ok=True)

        self.curr_img = 0
        self.mapping = dict()
# '/home/azhuavlev/Desktop/Data/Interhand_masked/annotations/test/'
        with open(self.base_folder + f'/annotations/{self.split}/InterHand2.6M_{self.split}_MANO_NeuralAnnot.json',
                  'r') as f:
            self.mano_dict = json.load(f)

        self.max_images_per_camera = max_images_per_camera

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
        n_images_per_camera = self.check_camera_img_count()
        n_images = min(n_images_per_camera, self.max_images_per_camera)

        for j_image in tqdm(range(n_images)):
            for i_camera, camera in enumerate(self.cameras_list):
                for image_class in ['masked', 'segmentations', 'mono_depth','images']:
                    camera_folder = self.base_folder + '/' + image_class + '/' + self.split + '/' + \
                                    f"Capture{self.capture_n}" + '/' + self.pose + '/' + f'cam{camera}'
                    img = sorted(os.listdir(camera_folder))[j_image]

                    grayscale = image_class in ['segmentations', 'mono_depth']

                    self.copy_image(
                        from_path=camera_folder,
                        img_name=img,
                        to_path=self.target_folder + '/' + image_class,
                        grayscale=grayscale)

                    self.create_mano(img)

                # camera_folder = self.pose_path + '/' + f'cam{camera}'
                # img = sorted(os.listdir(camera_folder))[j_image]
                #
                # self.copy_image(img, camera_folder)
                self.update_mapping(camera, i_camera, img)
                self.curr_img += 1

        self.save_mapping()

    def copy_image(self, from_path, img_name, to_path, grayscale):
        os.makedirs(to_path, exist_ok=True)
        # copy image to rgb folder
        os.system(f'cp {from_path}/{img_name} {to_path}/{img_name}')


        img_jpg = Image.open(f'{to_path}/{img_name}')

        if grayscale:
            img_jpg = img_jpg.convert('L')

        # save image as png and remove jpg
        img_jpg.save(f'{to_path}/{self.curr_img:05d}.png')
        os.system(f'rm {to_path}/{img_name}')

        # rename copied image to curr_img, padded to 5 digits
        # os.system(f'mv {self.rgb_path}/{img} {self.rgb_path}/{self.curr_img:05d}.jpg')

    def create_mano(self, img_name):
        img_id = img_name[5:-4]
        mano_params = self.mano_dict[self.capture_n][img_id]
        with open(f'{self.mano_path}/{self.curr_img:05d}.json', 'w') as f:
            json.dump(mano_params, f)


    def update_mapping(self, old_camera_id, new_camera_id, img):
        # self.mapping[f'{self.curr_img:05d}'] = {
        #     'camera': camera,
        #     # write img as old_img_id, with stemmed extension
        #     'old_img_id': os.path.splitext(img)[0],
        # }
        if new_camera_id not in self.mapping:
            self.mapping[new_camera_id] = {
                'old_camera_id': old_camera_id,
                'images_list': [],
            }
        self.mapping[new_camera_id]['images_list'].append({
            'old_img_name': img,
            'new_img_name': f'{self.curr_img:05d}.png'
        })

    def save_mapping(self):
        with open(f'{self.target_folder}/mapping.json', 'w') as f:
            json.dump(self.mapping, f)




if __name__ == '__main__':
    converter = InterhandToNeumanConverter(
        basefolder='/home/azhuavlev/Desktop/Data/Interhand_masked',
        split='test',
        capture_n='0',
        pose='ROM04_LT_Occlusion',
        # frames_list_slice=slice(None, 100, None),
        cameras_list=None,#['400262', '400263', '400264', '400265', '400284'],
        # val_cameras_frac=0.1,
        # max_cameras=15,
        experiment_n='02',
        max_images_per_camera=1,
        max_cameras=60
    )
    converter.copy_images()

    exit(0)
    # make video
    # print(pathlib.Path(converter.rgb_path))
    # imgs = make_video.load_all_images_from_dir(pathlib.Path(converter.rgb_path), 'png')
    # print(len(imgs))
    # concatenated_imgs = make_video.concatenate_images(imgs[:len(imgs) // 2], imgs[len(imgs) // 2:])
    # print(len(concatenated_imgs))
    # make_video.save_video(pathlib.Path(converter.target_folder), concatenated_imgs, fps=5)
