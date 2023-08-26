import torch
import kornia.morphology as kmorph

def sampling_mask_lpips(silhouettes):
    # only one image at a time
    assert silhouettes.shape[0] == 1

    patch_size = 64

    # dilate silhouettes to cover border regions
    sil_dilated = kmorph.dilation(
        silhouettes.unsqueeze(1), kernel=torch.ones(25, 25, device=silhouettes.device))

    # shift silhouettes up and left by patch_size // 2
    sil_shifted_up = torch.zeros_like(silhouettes[0])
    sil_shifted_up[:-patch_size // 2, :-patch_size // 2] = sil_dilated[0, 0, patch_size // 2:, patch_size // 2:]

    # find intersection of sil_dilated and sil_shifted_up
    sil_intersection = sil_dilated[0, 0] * sil_shifted_up

    # set pixels between right border and patch_size to 0, and same for bottom border
    sil_intersection[:, -patch_size:] = 0.0
    sil_intersection[-patch_size:, :] = 0.0

    # erode intersection and get border regions
    sil_intersection_usq = sil_intersection.unsqueeze(0).unsqueeze(0)
    eroded_kornia_0_25 = sil_intersection_usq - kmorph.erosion(
        sil_intersection_usq, kernel=torch.ones(25, 25, device=silhouettes.device))
    eroded_kornia_25_end = sil_intersection_usq - eroded_kornia_0_25

    # assign weight of 3 to border regions, 1 to inside regions
    full_mask_kornia = 3.0 * eroded_kornia_0_25 + 1.0 * eroded_kornia_25_end
    full_mask_kornia = full_mask_kornia.squeeze(0).squeeze(0)

    # flatten mask and sample top left corner
    full_mask_kornia_reshaped = full_mask_kornia.reshape(1, -1)
    top_left_idx = full_mask_kornia_reshaped.multinomial(1)

    # convert top left corner index to 2d
    top_left_idx_2d = torch.tensor([top_left_idx // full_mask_kornia.shape[1],
                                    top_left_idx % full_mask_kornia.shape[1]], device=silhouettes.device)

    # create patch mask by setting patch_size x patch_size region to 1
    patch_mask = torch.zeros_like(full_mask_kornia)
    patch_mask[
        top_left_idx_2d[0]:top_left_idx_2d[0] + patch_size,
        top_left_idx_2d[1]:top_left_idx_2d[1] + patch_size
    ] = 1

    # make sure we selected only patch_size x patch_size region
    assert patch_mask.sum() == patch_size ** 2, f'patch_mask.sum() = {patch_mask.sum()}, patch_size ** 2 = {patch_size ** 2}'

    return patch_mask.unsqueeze(0)

