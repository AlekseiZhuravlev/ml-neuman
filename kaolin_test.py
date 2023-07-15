import json
import numpy as np
# import matplotlib.pyplot as plt
import pickle
# from torch import inf
import sys
import torch
from kaolin.ops.mesh import check_sign
from kaolin.ops.mesh import index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from torch import inf

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    verts = torch.tensor([[[0., 0., 0.],
                           [1., 0.5, 1.],
                           [0.5, 1., 1.],
                           [1., 1., 0.5]]], device=device)
    faces = torch.tensor([[0, 3, 1],
                          [0, 1, 2],
                          [0, 2, 3],
                          [3, 2, 1]], device=device)
    axis = torch.linspace(0.1, 0.9, 3, device=device)
    p_x, p_y, p_z = torch.meshgrid(axis + 0.01, axis + 0.02, axis + 0.03)
    points = torch.cat((p_x.unsqueeze(-1), p_y.unsqueeze(-1), p_z.unsqueeze(-1)), dim=3)
    points = points.view(1, -1, 3)
    print('check_sign', check_sign(verts, faces, points))

    point = torch.tensor([[[0.5, 0.5, 0.5],
                           [3., 4., 5.]]], device='cuda')
    vertices = torch.tensor([[[0., 0., 0.],
                              [0., 1., 0.],
                              [0., 0., 1.]]], device='cuda')
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device='cuda')
    face_vertices = index_vertices_by_faces(vertices, faces)
    distance, index, dist_type = point_to_mesh_distance(point, face_vertices)
    print('point_to_mesh_distance', distance, index, dist_type)