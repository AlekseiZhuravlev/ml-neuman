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

def render_point_cloud(verts, features, cameras, no_grad, background_color):
    device = verts.device

    mesh = Pointclouds(points=verts, features=features)

    raster_settings = PointsRasterizationSettings(
        image_size=(512, 334),
        radius=0.01,
        points_per_pixel=10
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=background_color)
    )
    # render
    if no_grad:
        with torch.no_grad():
            images_depths = renderer(mesh)
    else:
        images_depths = renderer(mesh)

    depthmaps = images_depths[:, :, :, 3].unsqueeze(-1)
    images = images_depths[:, :, :, :3] # * 255

    return images, depthmaps