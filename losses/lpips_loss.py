from utils.constant import HARD_SURFACE_OFFSET, PATCH_SIZE, PATCH_SIZE_SQUARED, CANONICAL_ZOOM_FACTOR, CANONICAL_CAMERA_DIST
import torch

def lpips_loss(lpips_loss_fn, fine_rgb_map, batch_color, penalize_lpips):
    temp_lpips_loss = lpips_loss_fn(
        fine_rgb_map[:PATCH_SIZE_SQUARED].reshape(PATCH_SIZE, PATCH_SIZE, -1).permute(2, 0, 1) * 2 - 1,
        batch_color[:PATCH_SIZE_SQUARED].reshape(PATCH_SIZE, PATCH_SIZE, -1).permute(2, 0,
                                                                                        1) * 2 - 1
    ) * penalize_lpips
    assert torch.numel(temp_lpips_loss) == 1

    return temp_lpips_loss.flatten()[0]