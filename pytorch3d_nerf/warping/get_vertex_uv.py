from pytorch3d.io import load_obj
import torch

class VertexUVFinder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('faces_uvs_index', self.build_faces_uv_index())

    def get_point_uv(self, points_bary, face_ids):

        closest_face_uvs = self.faces_uvs_index[face_ids]
        # perform barycentric interpolation
        points_uvs = torch.einsum('ijk,ij->ik', closest_face_uvs, points_bary)

        return points_uvs

    def build_faces_uv_index(self):
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

