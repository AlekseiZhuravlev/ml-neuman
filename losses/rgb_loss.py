import torch
import torch.nn.functional as F
from utils import render_utils

def rgb_loss(fine_rgb_map, batch_is_hit, batch_color):

    # get which rays hit the object
    hit_index = torch.nonzero(batch_is_hit)[:, 0]

    return F.mse_loss(fine_rgb_map[hit_index], batch_color[hit_index])
