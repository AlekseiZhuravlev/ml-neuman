import torch
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    PointLights,
    DirectionalLights,
    PerspectiveCameras,
    Materials,
    SoftPhongShader,
    RasterizationSettings,
    MeshRenderer,
    MeshRendererWithFragments,
    MeshRasterizer,
    TexturesVertex,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
    PointsRasterizationSettings,
)

def render_point_cloud(verts, features, cameras):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    mesh = Pointclouds(points=verts, features=features)

    raster_settings = PointsRasterizationSettings(
        image_size=(512, 334),
        radius=0.01,
        points_per_pixel=10
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(1, 1, 1))
    )
    # render
    with torch.no_grad():
        images = renderer(mesh)
        images = images[:, :, :, :3] # * 255
        depthmaps = images.max(dim=-1)[0].unsqueeze(-1)

    return images, depthmaps