def make_sampling_mask(silhouettes):
    # replace zeros with 0.1
    silhouettes = silhouettes + 0.1
    return silhouettes