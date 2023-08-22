# find closest point on mesh
# which distance is correct? maybe implementation of _PointFaceDistanceCustom is different.
# We choose _point_to_tri_distance_batched
# todo: get barycentric coordinates of closest point in closest face
# todo: get uv coords of closest face
# todo: get uv coords of closest point on mesh - barycentric interpolation
# todo: check if pts are inside mesh, multiply dist by -1 if not
import torch
import kaolin
import warping.kaolin_test as kaolin_test
from mano_custom import mano_pytorch3d
import trimesh
import warping.get_point_mesh_face_distance as get_point_mesh_face_distance
import pytorch3d
import warping.get_closest_point_from_triangle as get_closest_point_from_triangle
import warping.get_closest_point_on_mesh_github as get_closest_point_on_mesh_github
import warping.get_vertex_uv as get_vertex_uv
from pytorch3d.io import load_obj
import numpy as np
import warping.test_barycentric as test_barycentric
# import warping.warp_points as warp_points_base

import igl



class WarpUV(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # test kaolin
        print('test kaolin inside WarpUV')
        kaolin_test.test_kaolin()

        print('test barycentric_py3d')
        test_barycentric.test_closest_point_finding()

        # make mesh watertight by adding more faces to wrist area
        hand_model = mano_pytorch3d.create_mano_custom(return_right_hand=False)

        faces_zero_pose = torch.from_numpy(hand_model.faces.astype(np.int32))[None, :, :]
        verts_zero_pose = hand_model.get_flat_hand_vertices_pytorch3d('cpu')

        self.register_buffer("faces_unrepaired", faces_zero_pose.clone())

        mesh = trimesh.Trimesh(vertices=verts_zero_pose[0], faces=faces_zero_pose[0])
        boundary_vertices = [92, 38, 122, 118, 117, 119, 120, 108, 79, 78, 121, 214, 215, 279, 239, 234][::-1]

        centroid = mesh.vertices[boundary_vertices].mean(axis=0)
        vertices_repaired = np.append(verts_zero_pose[0], centroid[None, :], axis=0)

        new_faces = []
        for i in range(len(boundary_vertices) - 1):
            next_i = i + 1 % len(boundary_vertices)
            new_faces.append([len(vertices_repaired) - 1, boundary_vertices[i], boundary_vertices[next_i]])
        new_faces.append(
            [len(vertices_repaired) - 1, boundary_vertices[len(boundary_vertices) - 1], boundary_vertices[0]])

        self.register_buffer('faces_repaired',
                             torch.tensor(np.append(faces_zero_pose[0], new_faces, axis=0)).unsqueeze(0)
                             )

        mesh_repaired_trimesh = trimesh.Trimesh(vertices=vertices_repaired, faces=self.faces_repaired.squeeze(0).numpy())
        assert mesh_repaired_trimesh.is_watertight, 'mesh_repaired_trimesh is not watertight'

        self.uv_finder = get_vertex_uv.VertexUVFinder()

    def repair_vertices(self, vertices):

        boundary_vertices = [92, 38, 122, 118, 117, 119, 120, 108, 79, 78, 121, 214, 215, 279, 239, 234][::-1]

        vertices = vertices.reshape(-1, 3)
        centroid = vertices[boundary_vertices].mean(axis=0)

        vertices_repaired = torch.cat([vertices, centroid[None, :]], dim=0).unsqueeze(0)
        return vertices_repaired

    def warp_points(self, rays_points_world, rays_directions_world, vertices_posed, Ts):

        orig_shape = rays_points_world.shape

        rays_points_world = rays_points_world.reshape(1, -1, 3)

        vertices_posed_repaired = self.repair_vertices(vertices_posed)

        mesh_not_repaired = pytorch3d.structures.Meshes(
            verts=vertices_posed,
            faces=self.faces_unrepaired,
        )
        # distances, closest_points, closest_pts_uv = self.project_point_to_mesh(
        #     mesh_not_repaired,
        #     rays_points_world,
        #     vertices_posed_repaired,
        #     self.faces_repaired[0],
        # )
        distances, closest_points, closest_pts_uv = self.test_pytorch3d_closest_pt_batched(
            mesh_not_repaired,
            rays_points_world,
            vertices_posed_repaired,
            self.faces_repaired[0],
        )

        warped_points = torch.cat([closest_pts_uv.unsqueeze(0), distances.unsqueeze(-1)], dim=-1)

        warped_points = warped_points.reshape(orig_shape)
        return warped_points, rays_directions_world

    def clip_bary(self, points_bary):

        points_bary_clip = torch.clamp(points_bary, min=0.0)

        points_bary_sum = torch.clamp(points_bary_clip.sum(dim=-1), min=1e-5)
        points_bary_clip = torch.clamp(
            points_bary_clip / points_bary_sum.unsqueeze(-1),
            min=0.0,
            max=1.0,
        )
        return points_bary_clip


    ####################################################################
    # Testing
    ####################################################################

    def test_pytorch3d_closest_pt_batched(self, mesh, points, vertices_repaired, faces_repaired):
        device = points.device

        # get vertices/faces, index vertices by faces
        vertices_unrepaired = mesh.verts_packed().unsqueeze(0)
        faces_unrepaired = mesh.faces_packed()
        face_vertices_unrepaired = kaolin.ops.mesh.index_vertices_by_faces(vertices_unrepaired, faces_unrepaired)

        # kaolin distance to mesh (squared), indices of closest faces
        distance_kaolin, face_index_kaolin, dist_type_kaolin = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            points, face_vertices_unrepaired)

        # closest points from closest faces, distances to closest points
        distances_v2, closest_points = get_closest_point_from_triangle.TestPointMeshDistance._point_to_tri_distance_batched(
            points[0],
            mesh.verts_packed()[mesh.faces_packed()[face_index_kaolin[0]]]
        )
        # barycentric coordinates of closest points
        closest_points_bary = get_closest_point_from_triangle.TestPointMeshDistance._point_to_bary_batched(
            closest_points,
            mesh.verts_packed()[mesh.faces_packed()[face_index_kaolin[0]]]
        )
        # find points inside repaired mesh
        is_inside = kaolin.ops.mesh.check_sign(vertices_repaired, faces_repaired, points)

        # uv coordinates of closest points
        closest_pts_uv = self.uv_finder.get_point_uv(
            self.clip_bary(closest_points_bary),
            face_index_kaolin[0]
        )

        # multiply distances with -1 if point is inside
        distances_v2_signed = distances_v2.abs().sqrt() * (is_inside.int() * -2 + 1)


        if not ((closest_points_bary >= -1e-1).all() and (closest_points_bary <= 1 + 1e-1).all()):
            print('closest_points_bary', closest_points_bary[(closest_points_bary < -1e-1) | (closest_points_bary > 1 + 1e-1)])

            # igl distance to mesh (signed unsquared), indices of closest faces, closest points
            distances_igl, closest_faces_igl, closest_pts_igl = igl.signed_distance(
                points.cpu().numpy()[0],
                vertices_unrepaired.cpu().numpy()[0],
                faces_unrepaired.cpu().numpy(),
            )
            # igl barycentric coordinates of closest points (using points/faces found by my methods)
            closest_points_bary_igl = igl.barycentric_coordinates_tri(
                closest_points.detach().cpu().numpy().astype(np.float64),
                mesh.verts_packed()[mesh.faces_packed()[face_index_kaolin[0]]].detach().cpu().numpy()[:, 0, :].astype(
                    np.float64),
                mesh.verts_packed()[mesh.faces_packed()[face_index_kaolin[0]]].detach().cpu().numpy()[:, 1, :].astype(
                    np.float64),
                mesh.verts_packed()[mesh.faces_packed()[face_index_kaolin[0]]].detach().cpu().numpy()[:, 2, :].astype(
                    np.float64),
            )
            # check that barycentric coordinates are in [0, 1] range
            # assert (closest_points_bary >= -1e-1).all() and (closest_points_bary <= 1 + 1e-1).all()

            closest_points_bary = closest_points_bary.clamp(0, 1)

            # assert (is_inside.int().sum() - (
            #             torch.from_numpy(distances_igl) < 0).int().sum()).abs() < 10, 'n of pts inside, kaolin vs igl, ' \
            #                                                                           f'{is_inside.int().sum(), (torch.from_numpy(distances_igl) < 0).int().sum()}'
            if not ((is_inside.int().sum() - (
                        torch.from_numpy(distances_igl) < 0).int().sum()).abs() < 10):
                print('is_inside', f'{is_inside.int().sum(), (torch.from_numpy(distances_igl) < 0).int().sum()}')

            # test that closest points are the same as ground truth from igl
            # assert (closest_points - torch.tensor(closest_pts_igl).to(
            #     device)).abs().sum() < 0.5, 'closest_points, my vs igl, ' \
            #                                 f'{(closest_points - torch.tensor(closest_pts_igl).to(device)).abs().sum()}'
            if not ((closest_points - torch.tensor(closest_pts_igl).to(
                device)).abs().sum() < 2):
                print('closest_points', (closest_points - torch.tensor(closest_pts_igl).to(device)).abs().sum())

            # test that barycentric coordinates are the same as ground truth from igl
            distance_bary_py3d_igl = (closest_points_bary - torch.from_numpy(closest_points_bary_igl).to(
                closest_points_bary.device)).abs().sum(dim=-1)
            # assert distance_bary_py3d_igl.sum() < 0.5, 'distance_bary_py3d - bary_igl, ' \
            #                                            f'{distance_bary_py3d_igl.sum()}'
            if not (distance_bary_py3d_igl.sum() < 2):
                print('distance_bary_py3d_igl', distance_bary_py3d_igl.sum())
                wrong_bary = closest_points_bary[(distance_bary_py3d_igl > 0.1).nonzero()]
                print('wrong_bary', wrong_bary)
                wrong_bary_igl = torch.from_numpy(closest_points_bary_igl)[(distance_bary_py3d_igl > 0.1).nonzero().cpu()]
                print('wrong_bary_igl', wrong_bary_igl)

            # test that distances point - triangle are same as kaolin distances to mesh
            # assert (distances_v2.cpu() - distance_kaolin[
            #     0].cpu()).abs().sum() < 0.5, 'difference distances_v2 - distance_kaolin, ' \
            #                                  f'{(distances_v2.cpu() - distance_kaolin[0].cpu()).abs().sum()}'
            if not ((distances_v2.cpu() - distance_kaolin[
                0].cpu()).abs().sum() < 2):
                print('distances_v2', (distances_v2.cpu() - distance_kaolin[
                    0].cpu()).abs().sum())

            # test that signed distances to point -> closest point are same as distances_v2_signed
            dist_to_closest_points = (closest_points - points).norm(dim=-1)
            dist_to_closest_points_signed = dist_to_closest_points * (is_inside.int() * -2 + 1)
            # assert (distances_v2_signed.cpu() - dist_to_closest_points_signed.cpu()).abs().sum() < 0.5, \
            #     'difference distances_v2 - distances_to_closest_points, ' \
            #     f'{(distances_v2_signed.cpu() - dist_to_closest_points_signed.cpu()).abs().sum()}'
            if not ((distances_v2_signed.cpu() - dist_to_closest_points_signed.cpu()).abs().sum() < 2):
                print('distances_v2_signed', (distances_v2_signed.cpu() - dist_to_closest_points_signed.cpu()).abs().sum())

            # print how large is the difference between igl and pytorch3d
            # dist_to_closest_points_igl = (torch.from_numpy(closest_pts_igl) - points.cpu()).norm(dim=-1)
            # print('difference dist_to_closest_points_igl and igl',
            #       (dist_to_closest_points_igl[0].abs() - torch.from_numpy(distances_igl).abs()).abs().sum())
            # print('distances_igl - distance_kaolin',
            #       (torch.tensor(distances_igl).abs() - distance_kaolin[0].cpu().sqrt()).abs().sum())
            # print('difference distances_v2 - igl',
            #       (distances_v2_signed.cpu() - torch.from_numpy(distances_igl)).abs().sum())

        return distances_v2_signed, closest_points, closest_pts_uv













    ####################################################################
    # Actual code
    ####################################################################


    def project_point_to_mesh(self, mesh, points, vertices_repaired, faces_repaired):
        # get vertices/faces, index vertices by faces
        vertices_unrepaired = mesh.verts_packed().unsqueeze(0)
        faces_unrepaired = mesh.faces_packed()
        face_vertices_unrepaired = kaolin.ops.mesh.index_vertices_by_faces(vertices_unrepaired, faces_unrepaired)

        # kaolin distance to mesh (squared), indices of closest faces
        distance_kaolin, face_index_kaolin, dist_type_kaolin = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            points, face_vertices_unrepaired)

        # closest points from closest faces, distances to closest points
        distances_v2, closest_points = get_closest_point_from_triangle.TestPointMeshDistance._point_to_tri_distance_batched(
            points[0],
            mesh.verts_packed()[mesh.faces_packed()[face_index_kaolin[0]]]
        )
        # barycentric coordinates of closest points
        closest_points_bary = get_closest_point_from_triangle.TestPointMeshDistance._point_to_bary_batched(
            closest_points,
            mesh.verts_packed()[mesh.faces_packed()[face_index_kaolin[0]]]
        )
        print('closest_points_bary', closest_points_bary.shape)
        exit(0)
        # check that barycentric coordinates are in [0, 1] range
        assert (closest_points_bary >= -1.5e-1).all() and (closest_points_bary <= 1 + 1.5e-1).all(),\
            f'{closest_points_bary[(closest_points_bary < -1e-1) | (closest_points_bary > 1 + 1e-1)]}'

        closest_points_bary = closest_points_bary.clamp(0, 1)

        # find points inside repaired mesh
        is_inside = kaolin.ops.mesh.check_sign(vertices_repaired, faces_repaired, points)

        # uv coordinates of closest points
        # uv_finder = get_vertex_uv.VertexUVFinder().to(closest_points_bary.device)
        closest_pts_uv = self.uv_finder.get_point_uv(closest_points_bary, face_index_kaolin[0])

        # multiply distances with -1 if point is inside
        distances_v2_signed = distances_v2.abs().sqrt() * (is_inside.int() * -2 + 1)

        return distances_v2_signed, closest_points, closest_pts_uv


    def warp_points_batched(
            self,
            rays_points_world,
            rays_directions_world,
            vertices_posed,
            Ts,
            n_batches: int = 16,
        ):
            """
            Args:
                ray_points: torch.Size([1, 8192, 1, 32, 3])
                ray_directions: torch.Size([1, 8192, 1, 32, 3])
                n_batches: int = 16
            """
            batches_ray_points = torch.chunk(rays_points_world, chunks=n_batches, dim=1)
            batches_ray_dirs = torch.chunk(rays_directions_world, chunks=n_batches, dim=1)

            # For each batch, execute the standard forward pass and concatenate
            can_pts_py3d = torch.tensor([], device=rays_points_world.device)
            can_dirs_py3d = torch.tensor([], device=rays_points_world.device)

            for batch_idx in range(len(batches_ray_points)):
                can_pts_py3d_batch, can_dirs_py3d_batch = self.warp_points(
                    rays_points_world=batches_ray_points[batch_idx],
                    rays_directions_world=batches_ray_dirs[batch_idx],
                    vertices_posed=vertices_posed,
                    Ts=Ts,
                )
                can_pts_py3d = torch.cat([can_pts_py3d, can_pts_py3d_batch], dim=1)
                can_dirs_py3d = torch.cat([can_dirs_py3d, can_dirs_py3d_batch], dim=1)

            return can_pts_py3d, can_dirs_py3d


if __name__=='__main__':
    cl = WarpUV()

