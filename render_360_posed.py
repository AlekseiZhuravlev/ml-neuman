def main_posed_360(opt):
    scene = neuman_helper.NeuManReader.read_scene(
        opt.scene_dir,
        tgt_size=opt.render_size,
        normalize=opt.normalize,
        bkg_range_scale=opt.bkg_range_scale,
        human_range_scale=opt.human_range_scale,
        smpl_type='optimized'
    )
    if opt.geo_threshold < 0:
        bones = []
        for i in range(len(scene.captures)):
            bones.append(np.linalg.norm(scene.smpls[i]['joints_3d'][3] - scene.smpls[i]['joints_3d'][0]))
        opt.geo_threshold = np.mean(bones)
    net = human_nerf.HumanNeRF(opt)
    weights = torch.load(opt.weights_path, map_location='cpu')
    utils.safe_load_weights(net, weights['hybrid_model_state_dict'])

    cap_id = 50
    center, up = utils.smpl_verts_to_center_and_up(scene.verts[cap_id])
    dist = opt.geo_threshold # camera distance depends on the human size
    render_poses = render_utils.default_360_path(center, up, dist, opt.trajectory_resolution)

    for i, rp in enumerate(render_poses):
        can_cap = ResizedPinholeCapture(
            scene.captures[0].pinhole_cam,
            rp,
            tgt_size=scene.captures[0].size
        )
        out = render_utils.render_smpl_nerf(
            net,
            can_cap,
            scene.verts[cap_id],
            scene.faces,
            scene.Ts[cap_id],
            rays_per_batch=opt.rays_per_batch,
            samples_per_ray=opt.samples_per_ray,
            white_bkg=opt.white_bkg,
            render_can=False,
            geo_threshold=opt.geo_threshold
        )
        save_path = os.path.join('./demo', f'posed_360/{os.path.basename(opt.scene_dir)}', f'out_{str(i).zfill(4)}.png')
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        imageio.imsave(save_path, out)
        print(f'image saved: {save_path}')
