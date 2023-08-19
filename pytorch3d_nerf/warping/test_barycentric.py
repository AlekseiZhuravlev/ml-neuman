import warping.get_point_mesh_face_distance as get_point_mesh_face_distance
import pytorch3d
import warping.get_closest_point_from_triangle as get_closest_point_from_triangle
import warping.get_closest_point_on_mesh_github as get_closest_point_on_mesh_github
import warping.get_vertex_uv as get_vertex_uv
import warping.barycentric_py3d as barycentric_py3d

from pytorch3d.io import load_obj
import pytorch3d
import igl


def test_pytorch3d_closest_pt(mesh, points):
    distances, face_ids = get_point_mesh_face_distance.point_mesh_face_distance(
        mesh,
        pytorch3d.structures.Pointclouds(points=points)
    )
    closest_pts = []
    distances_v2 = []
    for i in range(points.shape[1]):
        dist_v2, closest_point = get_closest_point_from_triangle.TestPointMeshDistance._point_to_tri_distance(
            points[0][i],
            mesh.verts_packed()[mesh.faces_packed()[face_ids[i]]]
        )
        closest_pts.append(closest_point)
        distances_v2.append(dist_v2)
    closest_pts = torch.stack(closest_pts)
    distances_v2 = torch.stack(distances_v2)

    # get index where (distances-distances_v2).abs() is max
    # indx_max = (distances-distances_v2).abs().argmax()
    # print('max diff', (distances-distances_v2).abs().max())
    # print('max diff index', indx_max)
    # print('distances', distances[indx_max])
    # print('distances_v2', distances_v2[indx_max])

    return distances_v2, closest_pts

def test_github_closest_pt(mesh, points):
    distances, face_ids = get_point_mesh_face_distance.point_mesh_face_distance(
        mesh,
        pytorch3d.structures.Pointclouds(points=points)
    )
    clspt_gh = get_closest_point_on_mesh_github.ProjectMesh2Point(mesh)
    closest_points, dist_v3 = clspt_gh(
        points,
        face_ids.unsqueeze(0),
    )
    return closest_points

def test_pytorch3d_closest_pt_batched(mesh, points, vertices_repaired, faces_repaired):
    distances, face_ids = get_point_mesh_face_distance.point_mesh_face_distance(
        mesh,
        pytorch3d.structures.Pointclouds(points=points)
    )
    distances_v2, closest_points = get_closest_point_from_triangle.TestPointMeshDistance._point_to_tri_distance_batched(
        points[0],
        mesh.verts_packed()[mesh.faces_packed()[face_ids]]
    )
    closest_points_bary = get_closest_point_from_triangle.TestPointMeshDistance._point_to_bary_batched(
        closest_points,
        mesh.verts_packed()[mesh.faces_packed()[face_ids]]
    )
    print(closest_points.detach().cpu().numpy().dtype)
    print(mesh.verts_packed()[mesh.faces_packed()[face_ids]].detach().cpu().numpy()[:, 0, :].dtype)
    closest_points_bary_igl = igl.barycentric_coordinates_tri(
        closest_points.detach().cpu().numpy().astype(np.float64),
        mesh.verts_packed()[mesh.faces_packed()[face_ids]].detach().cpu().numpy()[:, 0, :].astype(np.float64),
        mesh.verts_packed()[mesh.faces_packed()[face_ids]].detach().cpu().numpy()[:, 1, :].astype(np.float64),
        mesh.verts_packed()[mesh.faces_packed()[face_ids]].detach().cpu().numpy()[:, 2, :].astype(np.float64),
    )

    closest_points_bary_unbatched = []
    for i in range(closest_points.shape[0]):
        closest_points_bary_unbatched.append(
            get_closest_point_from_triangle.TestPointMeshDistance._point_to_bary(
                closest_points[i],
                mesh.verts_packed()[mesh.faces_packed()[face_ids[i]]]
            )
        )
    closest_points_bary_unbatched = torch.stack(closest_points_bary_unbatched)

    closest_points_bary_trimesh = trimesh.triangles.points_to_barycentric(
        mesh.verts_packed()[mesh.faces_packed()[face_ids]].detach().cpu().numpy(),
        closest_points.detach().cpu().numpy(),
        method='cross'
    )

    # import pytorch3d


    closest_points_bary_another_py3d = []
    for i in range(closest_points.shape[0]):
        closest_points_bary_another_py3d.append(
            torch.tensor(
            barycentric_py3d.barycentric_coordinates(
                closest_points[i],
                mesh.verts_packed()[mesh.faces_packed()[face_ids[i]]][0],
                mesh.verts_packed()[mesh.faces_packed()[face_ids[i]]][1],
                mesh.verts_packed()[mesh.faces_packed()[face_ids[i]]][2],
            ),
                device = closest_points.device
            )
        )
    closest_points_bary_another_py3d = torch.stack(closest_points_bary_another_py3d)


    # assert that each barycentric coordinate is between 0 and 1
    assert (closest_points_bary >= -1e-1).all() and (closest_points_bary <= 1 + 1e-1).all()
    closest_points_bary = closest_points_bary.clamp(0, 1)

    print('closest_points_bary', closest_points_bary)
    print('closest_points_bary_igl', closest_points_bary_igl)
    print('closest_points_bary_trimesh', closest_points_bary_trimesh)

    distance_bary_py3d_igl = (closest_points_bary - torch.from_numpy(closest_points_bary_igl).to(closest_points_bary.device)).abs().sum(dim=-1)
    print('distance_bary_py3d_igl', distance_bary_py3d_igl.sum())
    distance_bary_batched_unbatched = (closest_points_bary - closest_points_bary_unbatched.to(closest_points_bary.device)).abs().sum(dim=-1).sum()
    print('distance_bary_batched_unbatched', distance_bary_batched_unbatched)
    distance_bary_py3d_trimesh = (closest_points_bary - torch.from_numpy(closest_points_bary_trimesh).to(closest_points_bary.device)).abs().sum(dim=-1)
    print('distance_bary_py3d_trimesh', distance_bary_py3d_trimesh.sum())
    distance_bary_py3d_another_py3d = (closest_points_bary - closest_points_bary_another_py3d.to(closest_points_bary.device)).abs().sum(dim=-1).sum()
    print('distance_bary_py3d_another_py3d', distance_bary_py3d_another_py3d.sum())
    distance_another_py3d_igl = (closest_points_bary_another_py3d - torch.from_numpy(closest_points_bary_igl).to(closest_points_bary.device)).abs().sum(dim=-1).sum()
    print('distance_another_py3d_igl', distance_another_py3d_igl.sum())
    distance_unbatched_igl = (closest_points_bary_unbatched.to(closest_points_bary.device) - torch.from_numpy(closest_points_bary_igl).to(closest_points_bary.device)).abs().sum(dim=-1).sum()
    print('distance_unbatched_igl', distance_unbatched_igl.sum())


    ind_wrong_bary = distance_bary_py3d_igl > 1e-2

    print('wrong bary igl', closest_points_bary_igl[ind_wrong_bary.cpu()])
    print('wrong bary py3d', closest_points_bary[ind_wrong_bary])

    print('points wrong bary', closest_points[ind_wrong_bary])
    print('vertices wrong bary', mesh.verts_packed()[mesh.faces_packed()[face_ids[ind_wrong_bary]]])

















    uv_finder = get_vertex_uv.VertexUVFinder().to(closest_points_bary.device)

    closest_pts_uv = uv_finder.get_point_uv(closest_points_bary, face_ids)

    is_inside = kaolin.ops.mesh.check_sign(vertices_repaired, faces_repaired, closest_points.unsqueeze(0))
    print('is_inside', is_inside.int().sum())
    # multiply distances with -1 if point is inside
    distances_v2 = distances_v2 * (is_inside.int() * -2 + 1)

    return distances_v2, closest_points, closest_pts_uv, closest_points_bary


if __name__ == '__main__':
    mesh_not_repaired = pytorch3d.structures.Meshes(
        verts=verts_zero_pose,
        faces=faces_zero_pose,
    )

    # points_to_query = pytorch3d.ops.sample_points_from_meshes(mesh_not_repaired, 1000000) * 2
    points_to_query = torch.rand(1, 5000, 3) * 0.5 - 0.25

    mesh_not_repaired = mesh_not_repaired.to(device)
    points_to_query = points_to_query.to(device)

    distances_batched, pts_batched, uvs_batched, closest_points_bary = test_pytorch3d_closest_pt_batched(
        mesh_not_repaired,
        points_to_query,
        torch.tensor(vertices_repaired).unsqueeze(0).to(device),
        torch.tensor(faces_repaired).to(device),
    )

    distances_unbatched, closest_pts_unbatched = test_pytorch3d_closest_pt(
        mesh_not_repaired,
        points_to_query,
    )

    # fixme: I am not entirely sure if everything is correct. Run experiments to check