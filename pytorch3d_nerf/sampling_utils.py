def make_sampling_mask(silhouettes):
    # replace zeros with 0.2
    silhouettes = silhouettes + 0.4
    return silhouettes