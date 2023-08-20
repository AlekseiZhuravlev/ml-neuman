from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from torch.autograd import Function
from torch.autograd.function import once_differentiable

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3

# PointFaceDistance
class _PointFaceDistanceCustom(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """

    @staticmethod
    def forward(ctx, points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_points: Scalar equal to maximum number of points in the batch
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.

            `dists[p]` is
            `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`

        """
        dists, idxs = _C.point_face_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
        )
        ctx.save_for_backward(points, tris, idxs)
        return dists, idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists
        )
        return grad_points, None, grad_tris, None, None


point_face_distance_custom = _PointFaceDistanceCustom.apply

def point_mesh_face_distance(meshes: Meshes, pcls: Pointclouds):
    """
    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds

    Returns:
        point_to_face_dist: A tensor of shape (P,) where P is the total number of
            points in the batch. The value `point_to_face_dist[p]` is the distance
            of the `p`-th point to the closest face in the corresponding mesh in the
            batch.
        closest_face_idxs: A tensor of shape (P,) where P is the total number of
            points in the batch. The value `closest_face_idxs[p]` is the index of
            the closest face in the corresponding mesh in the batch to the `p`-th
            point.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face_dist, closest_face_idxs = point_face_distance_custom(
        points, points_first_idx, tris, tris_first_idx, max_points, #1e-1
    )
    return point_to_face_dist, closest_face_idxs
