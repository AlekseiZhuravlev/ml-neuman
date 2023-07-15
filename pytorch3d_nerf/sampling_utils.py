import torch
def make_sampling_mask(silhouettes):
    # replace zeros with 0.1
    # TODO disabled
    silhouettes = silhouettes + 0.05
    # silhouettes = torch.ones_like(silhouettes)
    return silhouettes