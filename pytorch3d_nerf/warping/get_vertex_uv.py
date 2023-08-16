from pytorch3d.io import load_obj
import torch

def build_faces_uv_index():
    verts_from_obj, faces_from_obj, aux = load_obj(
        '/itet-stor/azhuavlev/net_scratch/Projects/Data/models/mano/uv_maps/MANO_UV_left.obj')

    faces_uvs_index = []
    for i_face in range(faces_from_obj.verts_idx.shape[0]):
        # face = faces_from_obj.verts_idx[i_face]
        face_texture_ids = faces_from_obj.textures_idx[i_face]
        face_uv_coords = aux.verts_uvs[face_texture_ids]
        faces_uvs_index.append(face_uv_coords)
    faces_uvs_index = torch.stack(faces_uvs_index)

    return faces_uvs_index

def get_point_uv(points_bary, face_ids):
    faces_uvs_index = build_faces_uv_index().to(points_bary.device)

    closest_face_uvs = faces_uvs_index[face_ids]
    # perform barycentric interpolation
    points_uvs = torch.einsum('ijk,ij->ik', closest_face_uvs, points_bary)

    # for i in range(points_bary.shape[0]):
    #     regular = points_bary[i][0] * closest_face_uvs[i][0] +\
    #               points_bary[i][1] * closest_face_uvs[i][1] +\
    #               points_bary[i][2] * closest_face_uvs[i][2]
    #     assert torch.allclose(regular, points_uvs[i])

    return points_uvs