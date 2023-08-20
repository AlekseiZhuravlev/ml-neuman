import os
import sys
import time
import json
import glob
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython import display
from tqdm.notebook import tqdm

# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)
import sys
import trimesh


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


def test_pytorch3d_closest_pt_batched(mesh, points, vertices_repaired, faces_repaired):

    # ground truth data from igl
    distances_igl, closest_faces_igl, closest_pts_igl = igl.signed_distance(
        points.cpu().numpy()[0],
        mesh.verts_packed().cpu().numpy(),
        mesh.faces_packed().cpu().numpy(),
    )

    # closest faces
    distances_point_face, face_ids = get_point_mesh_face_distance.point_mesh_face_distance(
        mesh,
        pytorch3d.structures.Pointclouds(points=points)
    )
    # closest points from closest faces
    distances_to_closest_pt, closest_points = get_closest_point_from_triangle.TestPointMeshDistance._point_to_tri_distance_batched(
        points[0],
        mesh.verts_packed()[mesh.faces_packed()[face_ids]]
    )
    print(((closest_points.cpu() - closest_pts_igl).abs() > 1e-2).int().sum())
    assert ((closest_points.cpu() - closest_pts_igl).abs() < 5e-2).all(), 'closest_points - closest_pts_igl'

    # closest points to barycentric coordinates
    closest_points_bary = get_closest_point_from_triangle.TestPointMeshDistance._point_to_bary_batched(
        closest_points,
        mesh.verts_packed()[mesh.faces_packed()[face_ids]]
    )
    # ground truth baricentric coordinates from igl
    closest_points_bary_igl = igl.barycentric_coordinates_tri(
        closest_points.detach().cpu().numpy().astype(np.float64),
        mesh.verts_packed()[mesh.faces_packed()[face_ids]].detach().cpu().numpy()[:, 0, :].astype(np.float64),
        mesh.verts_packed()[mesh.faces_packed()[face_ids]].detach().cpu().numpy()[:, 1, :].astype(np.float64),
        mesh.verts_packed()[mesh.faces_packed()[face_ids]].detach().cpu().numpy()[:, 2, :].astype(np.float64),
    )
    # assert that each barycentric coordinate is between 0 and 1
    assert (closest_points_bary >= -0.1).all() and (closest_points_bary <= 1.1).all()
    closest_points_bary = closest_points_bary.clamp(0, 1)

    assert ((closest_points_bary.cpu() - torch.from_numpy(closest_points_bary_igl)).abs() < 0.01).all(), 'distance_bary_py3d_igl'

    is_inside = kaolin.ops.mesh.check_sign(vertices_repaired, faces_repaired, points)
    assert is_inside.int().sum() > 0

    # multiply distances with -1 if point is inside
    distances_to_closest_pt_signed = distances_to_closest_pt * (is_inside.int() * -2 + 1)

    print(((distances_to_closest_pt_signed.cpu() - distances_igl).abs() > 1e-2).int().sum(),
          'distances_to_closest_pt_signed - distances_igl')
    assert ((distances_to_closest_pt_signed.cpu() - distances_igl).abs() < 5e-2).all(), 'distances_to_closest_pt_signed - distances_igl'


    uv_finder = get_vertex_uv.VertexUVFinder().to(closest_points_bary.device)
    closest_pts_uv = uv_finder.get_point_uv(closest_points_bary, face_ids)

    return distances_to_closest_pt_signed, closest_points, closest_pts_uv, closest_points_bary


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
    )

    # points_to_query = pytorch3d.ops.sample_points_from_meshes(mesh_not_repaired, 1000000) * 2
    points_to_query = torch.rand(1, 50000, 3) * 0.5 - 0.25

    mesh_not_repaired = mesh_not_repaired.to(device)
    points_to_query = points_to_query.to(device)

    distances_batched, pts_batched, uvs_batched, closest_points_bary = test_pytorch3d_closest_pt_batched(
        mesh_not_repaired,
        points_to_query,
        torch.tensor(vertices_repaired).unsqueeze(0).to(device),
        torch.tensor(faces_repaired).to(device),
    )
    print('all tests passed')

if __name__ == '__main__':
    test_closest_point_finding()
