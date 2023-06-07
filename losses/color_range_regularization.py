import torch
import torch.nn.functional as F

def color_range_regularization(coarse_human_net, pts, dirs, tgts, penalize_color_range):
    device = pts.device

    # create dummy directions with random values
    dummy_dirs = torch.randn(dirs.shape, dtype=dirs.dtype, device=device)
    dummy_dirs = dummy_dirs / torch.norm(dummy_dirs, dim=-1, keepdim=True)

    # compute the output of the network with the dummy directions
    dummy_out = coarse_human_net(pts, dummy_dirs)

    # penalize the color range of the output
    # by minimizing the MSE between the output and the target
    color_reg = F.mse_loss(
        torch.sigmoid(dummy_out.reshape(-1, 4))[:, :3],
        torch.sigmoid(tgts.reshape(-1, 4))[:, :3]
    ) * penalize_color_range

    return color_reg