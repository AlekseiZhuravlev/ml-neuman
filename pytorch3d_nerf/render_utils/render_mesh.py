import torch
from pytorch3d.structures import Meshes
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
TexturesVertex)

def render_mesh(mesh, face, cameras):
    device = mesh.device

    batch_size, vertex_num = mesh.shape[:2]

    textures = TexturesVertex(verts_features=torch.ones((batch_size, vertex_num, 3)).float().to(device))
    mesh = Meshes(mesh, face, textures)

    img_size = (512, 334)
    raster_settings = RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1,
                                            perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)
    lights = PointLights(device=device)
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    materials = Materials(
        device=device,
        specular_color=((0.0, 1.0, 0.0),),
        shininess=0
    )

    # render
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)
        images = images[:, :, :, :3] #* 255
        depthmaps = fragments.zbuf

    return images, depthmaps