import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import sys
import time
import torch
import trimesh
from IPython import display
from PIL import Image
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)
# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from tqdm.notebook import tqdm

sys.path.append('/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf')
from mano_custom import mano_pytorch3d

device = torch.device("cuda:0")
import kaolin
import warping.kaolin_test as kaolin_test

import warping.get_point_mesh_face_distance as get_point_mesh_face_distance
import pytorch3d
import warping.get_closest_point_from_triangle as get_closest_point_from_triangle
import warping.get_closest_point_on_mesh_github as get_closest_point_on_mesh_github
import warping.get_vertex_uv as get_vertex_uv
import warping.barycentric_py3d as barycentric_py3d

from pytorch3d.io import load_obj
import pytorch3d
import igl
from kaolin.ops.mesh import index_vertices_by_faces



def test_pytorch3d_closest_pt_batched(mesh, points, vertices_repaired, faces_repaired):

    # get vertices/faces, index vertices by faces
    vertices_unrepaired = mesh.verts_packed().unsqueeze(0)
    faces_unrepaired = mesh.faces_packed()
    face_vertices_unrepaired = kaolin.ops.mesh.index_vertices_by_faces(vertices_unrepaired, faces_unrepaired)

    # kaolin distance to mesh (squared), indices of closest faces
    distance_kaolin, face_index_kaolin, dist_type_kaolin = kaolin.metrics.trianglemesh.point_to_mesh_distance(points, face_vertices_unrepaired)

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
    # igl distance to mesh (signed unsquared), indices of closest faces, closest points
    distances_igl, closest_faces_igl, closest_pts_igl = igl.signed_distance(
        points.cpu().numpy()[0],
        vertices_unrepaired.cpu().numpy()[0],
        faces_unrepaired.cpu().numpy(),
    )
    # igl barycentric coordinates of closest points (using points/faces found by my methods)
    closest_points_bary_igl = igl.barycentric_coordinates_tri(
        closest_points.detach().cpu().numpy().astype(np.float64),
        mesh.verts_packed()[mesh.faces_packed()[face_index_kaolin[0]]].detach().cpu().numpy()[:, 0, :].astype(np.float64),
        mesh.verts_packed()[mesh.faces_packed()[face_index_kaolin[0]]].detach().cpu().numpy()[:, 1, :].astype(np.float64),
        mesh.verts_packed()[mesh.faces_packed()[face_index_kaolin[0]]].detach().cpu().numpy()[:, 2, :].astype(np.float64),
    )
    # check that barycentric coordinates are in [0, 1] range
    assert (closest_points_bary >= -1e-1).all() and (closest_points_bary <= 1 + 1e-1).all()
    closest_points_bary = closest_points_bary.clamp(0, 1)

    # find points inside repaired mesh
    is_inside = kaolin.ops.mesh.check_sign(vertices_repaired, faces_repaired, points)

    assert (is_inside.int().sum() - (torch.from_numpy(distances_igl) < 0).int().sum()).abs() < 10, 'n of pts inside, kaolin vs igl, ' \
            f'{is_inside.int().sum(), (torch.from_numpy(distances_igl) < 0).int().sum()}'

    # uv coordinates of closest points
    uv_finder = get_vertex_uv.VertexUVFinder().to(closest_points_bary.device)
    closest_pts_uv = uv_finder.get_point_uv(closest_points_bary, face_index_kaolin[0])

    # multiply distances with -1 if point is inside
    distances_v2_signed = distances_v2.abs().sqrt() * (is_inside.int() * -2 + 1)

    # test that closest points are the same as ground truth from igl
    assert (closest_points - torch.tensor(closest_pts_igl).to(device)).abs().sum() < 0.5, 'closest_points, my vs igl, ' \
            f'{(closest_points - torch.tensor(closest_pts_igl).to(device)).abs().sum()}'

    # test that barycentric coordinates are the same as ground truth from igl
    distance_bary_py3d_igl = (closest_points_bary - torch.from_numpy(closest_points_bary_igl).to(
        closest_points_bary.device)).abs().sum(dim=-1)
    assert distance_bary_py3d_igl.sum() < 0.5,'distance_bary_py3d - bary_igl, ' \
            f'{distance_bary_py3d_igl.sum()}'

    # test that distances point - triangle are same as kaolin distances to mesh
    assert (distances_v2.cpu() - distance_kaolin[0].cpu()).abs().sum() < 0.5, 'difference distances_v2 - distance_kaolin, ' \
            f'{(distances_v2.cpu() - distance_kaolin[0].cpu()).abs().sum()}'

    # test that signed distances to point -> closest point are same as distances_v2_signed
    dist_to_closest_points = (closest_points - points).norm(dim=-1)
    dist_to_closest_points_signed = dist_to_closest_points * (is_inside.int() * -2 + 1)
    assert (distances_v2_signed.cpu() - dist_to_closest_points_signed.cpu()).abs().sum() < 0.5,\
        'difference distances_v2 - distances_to_closest_points, ' \
        f'{(distances_v2_signed.cpu() - dist_to_closest_points_signed.cpu()).abs().sum()}'

    # print how large is the difference between igl and pytorch3d
    dist_to_closest_points_igl = (torch.from_numpy(closest_pts_igl) - points.cpu()).norm(dim=-1)
    print('difference dist_to_closest_points_igl and igl', (dist_to_closest_points_igl[0].abs() - torch.from_numpy(distances_igl).abs()).abs().sum())
    print('distances_igl - distance_kaolin', (torch.tensor(distances_igl).abs() - distance_kaolin[0].cpu().sqrt()).abs().sum())
    print('difference distances_v2 - igl', (distances_v2_signed.cpu() - torch.from_numpy(distances_igl)).abs().sum())

    return distances_v2_signed, closest_points, closest_pts_uv, closest_points_bary


def test_closest_point_finding():
    kaolin_test.test_kaolin()
    hand_model = mano_pytorch3d.create_mano_custom(return_right_hand=False)

    faces_zero_pose = torch.from_numpy(hand_model.faces.astype(np.int32))[None, :, :]
    verts_zero_pose = hand_model.get_flat_hand_vertices_pytorch3d('cpu')

    mesh = trimesh.Trimesh(vertices=verts_zero_pose[0], faces=faces_zero_pose[0])
    boundary_vertices = [92, 38, 122, 118, 117, 119, 120, 108, 79, 78, 121, 214, 215, 279, 239, 234][::-1]

    centroid = mesh.vertices[boundary_vertices].mean(axis=0)
    vertices_repaired = np.append(verts_zero_pose[0], centroid[None, :], axis=0)

    new_faces = []
    for i in range(len(boundary_vertices) - 1):
        next_i = i + 1 % len(boundary_vertices)
        new_faces.append([len(vertices_repaired) - 1, boundary_vertices[i], boundary_vertices[next_i]])
    new_faces.append([len(vertices_repaired) - 1, boundary_vertices[len(boundary_vertices) - 1], boundary_vertices[0]])

    faces_repaired = np.append(faces_zero_pose[0], new_faces, axis=0)

    mesh_repaired_trimesh = trimesh.Trimesh(vertices=vertices_repaired, faces=faces_repaired)
    assert mesh_repaired_trimesh.is_watertight

    mesh_not_repaired = pytorch3d.structures.Meshes(
        verts=verts_zero_pose,
        faces=faces_zero_pose,
    ).to(device)

    for i in range(10):
        points_to_query = torch.rand(1, 50000, 3, device=device) * 0.1 - 0.05
        distances_batched, pts_batched, uvs_batched, closest_points_bary = test_pytorch3d_closest_pt_batched(
            mesh_not_repaired,
            points_to_query,
            torch.tensor(vertices_repaired).unsqueeze(0).to(device),
            torch.tensor(faces_repaired).to(device),
        )
        print(f'****{i} test passed')
    print('******closest points finding: all tests passed')


if __name__ == '__main__':
    test_closest_point_finding()
