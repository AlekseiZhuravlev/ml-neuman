import torch
import torch.nn.functional as F


def _smpl_symmetry_regularization(coarse_human_net, pts, dirs, tgts, penalize_symmetric_alpha):
    '''
    we use dummy ray directions for the flipped points, since we only
    care about the occupancy symmetry.
    '''

    # flip the x coordinate of the points
    pts_flip = pts.clone().detach()
    pts_flip[..., 0] *= -1

    # compute the output of the network with the flipped points and their original directions
    out_flip = coarse_human_net(pts_flip, dirs.clone().detach())

    # penalize the symmetry of the output
    sym_reg = F.mse_loss(
        torch.tanh(torch.relu(tgts[..., 3])),
        torch.tanh(torch.relu(out_flip[..., 3]))
    ) * penalize_symmetric_alpha

    return sym_reg