import torch
import torch.nn as nn
from helpers import sample_images_at_mc_locs


class SilhouetteLossWorld(nn.Module):
    def __init__(self, loss_func, sil_loss_start_factor, sil_loss_epochs):
        super().__init__()
        self.loss_func = loss_func
        self.sil_loss_start_factor = sil_loss_start_factor
        self.sil_loss_epochs = sil_loss_epochs

    def forward(self,
                rendered_silhouettes,
                silhouettes,
                ray_bundle,
                current_epoch,
                ):

        # silhouettes_at_rays.shape rendered_silhouettes.shape torch.Size([1, 8192, 1, 1])
        silhouettes_at_rays = sample_images_at_mc_locs(
            silhouettes.unsqueeze(-1),
            ray_bundle.xys
        )

        sil_err_unconstrained = self.loss_func(
            rendered_silhouettes,
            silhouettes_at_rays,
        )

        # decrease silhouette loss and update the factor
        if self.sil_loss_epochs > 0:
            sil_loss_factor = self.sil_loss_start_factor * max(0, 1 - (current_epoch / self.sil_loss_epochs))
        else:
            sil_loss_factor = 0
        sil_err = sil_err_unconstrained * sil_loss_factor

        return sil_err, sil_err_unconstrained, sil_loss_factor
