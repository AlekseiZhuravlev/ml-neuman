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
# import warping.warp_points as warp_points_base



class WarpUV(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # test kaolin
        print('test kaolin inside WarpUV')
        kaolin_test.test_kaolin()

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
        distances, closest_points, closest_pts_uv = self.project_point_to_mesh(
            mesh_not_repaired,
            rays_points_world,
            vertices_posed_repaired
        )

        warped_points = torch.cat([closest_pts_uv.unsqueeze(0), distances.unsqueeze(-1)], dim=-1)

        warped_points = warped_points.reshape(orig_shape)
        return warped_points, rays_directions_world


    def project_point_to_mesh(self, mesh_unrepaired, points, vertices_repaired):
        distances, face_ids = get_point_mesh_face_distance.point_mesh_face_distance(
            mesh_unrepaired,
            pytorch3d.structures.Pointclouds(points=points)
        )
        distances_v2, closest_points = get_closest_point_from_triangle.TestPointMeshDistance._point_to_tri_distance_batched(
            points[0],
            mesh_unrepaired.verts_packed()[mesh_unrepaired.faces_packed()[face_ids]]
        )
        closest_points_bary = get_closest_point_from_triangle.TestPointMeshDistance._point_to_bary_batched(
            closest_points,
            mesh_unrepaired.verts_packed()[mesh_unrepaired.faces_packed()[face_ids]]
        )

        # assert that each barycentric coordinate is between 0 and 1
        assert (closest_points_bary >= -1e-4).all() and (closest_points_bary <= 1 + 1e-4).all()
        closest_points_bary = closest_points_bary.clamp(0, 1)

        closest_pts_uv = self.uv_finder.get_point_uv(closest_points_bary, face_ids)

        is_inside = kaolin.ops.mesh.check_sign(vertices_repaired, self.faces_repaired[0], points)

        # assert that some points are inside
        assert is_inside.sum() > 0

        # multiply distances with -1 if point is inside
        distances_v2 = distances_v2 * (is_inside.int() * -2 + 1)

        return distances_v2, closest_points, closest_pts_uv


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

