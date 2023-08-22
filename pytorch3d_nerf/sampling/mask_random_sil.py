import torch
import kornia.morphology as kmorph


# replace zeros with 0.1
# silhouettes = silhouettes + 0.3
# silhouettes = torch.ones_like(silhouettes)


def sampling_mask_0_25(silhouettes):

    silhouettes = silhouettes.unsqueeze(1)

    segment_border_0_kornia = torch.ones_like(silhouettes) - silhouettes
    eroded_kornia_0_25 = silhouettes - kmorph.erosion(
        silhouettes, kernel=torch.ones(25, 25, device=silhouettes.device))
    eroded_kornia_25_end = silhouettes - eroded_kornia_0_25

    full_mask_kornia = 0.3 * segment_border_0_kornia + \
                       3.0 * eroded_kornia_0_25 + \
                       1.0 * eroded_kornia_25_end

    return full_mask_kornia.squeeze(1)


def sampling_mask_25_0_25(silhouettes):

    silhouettes = silhouettes.unsqueeze(1)

    segment_border_0_kornia = torch.ones_like(silhouettes) - silhouettes

    dilated_kornia_25_0 = kmorph.dilation(
        silhouettes, kernel=torch.ones(25, 25, device=silhouettes.device)) - silhouettes
    segment_border_25_0_kornia = segment_border_0_kornia - dilated_kornia_25_0

    eroded_kornia_0_25 = silhouettes - kmorph.erosion(
        silhouettes, kernel=torch.ones(25, 25, device=silhouettes.device))
    eroded_kornia_25_end = silhouettes - eroded_kornia_0_25

    full_mask_kornia = 0.3 * segment_border_25_0_kornia + \
                        2.0 * dilated_kornia_25_0 + \
                       3.0 * eroded_kornia_0_25 + \
                       1.0 * eroded_kornia_25_end

    return full_mask_kornia.squeeze(1)