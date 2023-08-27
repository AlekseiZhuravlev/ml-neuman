import cv2
import numpy as np
from numpy.core.numeric import zeros_like


def grabcut_refine(mano_mask, rgb_img):
    # erode the MANO mask
    kernel = np.ones((25, 25), np.uint8)
    mano_mask_eroded = cv2.erode(mano_mask * 255, kernel, iterations=1)

    grabCut_mask = zeros_like(mano_mask)
    grabCut_mask[mano_mask_eroded > 0] = cv2.GC_PR_FGD
    grabCut_mask[mano_mask_eroded == 0] = cv2.GC_PR_BGD

    # GRABCUT
    # allocate memory for two arrays that the GrabCut algorithm internally uses when segmenting the foreground from the background
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    # apply GrabCut using the the mask segmentation method
    (mask, bgModel, fgModel) = cv2.grabCut(rgb_img, grabCut_mask, None, bgModel, fgModel, iterCount=20,
                                           mode=cv2.GC_INIT_WITH_MASK)

    # set all definite background and probable background pixels to 0 while definite foreground and probable foreground pixels are set to 1, then scale teh mask from the range [0, 1] to [0, 255]
    refined_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
    refined_mask = (refined_mask * 255).astype("uint8")
    refined_mask = refined_mask[..., 0]

    return refined_mask


def largest_component(mask):
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)],
                              key=lambda x: x[1])  # Note: range() starts from 1 since 0 is the background label.
    finalMask = zeros_like(mask)
    finalMask[labels == max_label] = 255
    return finalMask


def remove_forearm(mano_mask, mano_mask_refined):
    kernel = np.ones((10, 10), np.uint8)
    mano_mask_dilated = cv2.dilate(mano_mask, kernel, iterations=1)
    _, diff = cv2.threshold(mano_mask_refined - mano_mask_dilated, 127, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(diff) == 0:  # mano_mask_dilated encapsulates the mano_mask_refined; nothing to remove
        return mano_mask_refined

    probable_forearm = largest_component(diff)
    # estimate mask area
    mask_area_frac = cv2.countNonZero(probable_forearm) / (mano_mask.shape[0] * mano_mask.shape[1])

    if mask_area_frac > 0.01:
        # extra region big enough to be a forearm
        return mano_mask_refined - probable_forearm
    else:
        # its probably some part of the palm
        return mano_mask_refined