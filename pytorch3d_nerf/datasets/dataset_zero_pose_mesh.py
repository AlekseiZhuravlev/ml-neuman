from losses.canonical_utils.test_canonical_cameras import render_zero_pose
from losses.canonical_utils.cameras_canonical import create_canonical_cameras


if __name__ =='__main__':

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    n_cameras = 20
    cameras = create_canonical_cameras(n_cameras, random_cameras=False, device=device)

    # print(len(cameras))
    # exit()

    print(cameras.get_image_size())

    img_zero_pose, depth_zero_pose, verts_zpose = render_zero_pose(cameras)


