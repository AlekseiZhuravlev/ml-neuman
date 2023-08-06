import torch
import kornia.morphology as kmorph

def make_sampling_mask(silhouettes):
    # replace zeros with 0.1
    # silhouettes = silhouettes + 0.3
    # silhouettes = torch.ones_like(silhouettes)

    segment_border_0_kornia = torch.ones_like(silhouettes) - silhouettes
    eroded_kornia_0_25 = silhouettes - kmorph.erosion(
        silhouettes, kernel=torch.ones(25, 25, device=silhouettes.device))
    eroded_kornia_25_end = silhouettes - eroded_kornia_0_25

    full_mask_kornia = 0.3 * segment_border_0_kornia + \
                       3.0 * eroded_kornia_0_25 + \
                       1.0 * eroded_kornia_25_end

    return full_mask_kornia