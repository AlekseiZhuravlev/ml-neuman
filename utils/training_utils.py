

def densepose_name_to_idx():
    return {
        'Torso': [1, 2],
        'Right Hand': [3],
        'Left Hand': [4],
        'Left Foot': [5],
        'Right Foot': [6],
        'Upper Leg Right': [7, 9],
        'Upper Leg Left': [8, 10],
        'Lower Leg Right': [11, 13],
        'Lower Leg Left': [12, 14],
        'Upper Arm Left': [15, 17],
        'Upper Arm Right': [16, 18],
        'Lower Arm Left': [19, 21],
        'Lower Arm Right': [20, 22],
        'Head': [23, 24]
    }


def densepose_idx_to_name():
    name2idx = densepose_name_to_idx()
    idx2name = {}
    for k, v in name2idx.items():
        for item in v:
            idx2name[item] = k
    return idx2name


def turn_smpl_gradient_off(dp_mask):
    assert dp_mask is not None
    grad_mask = np.ones([24, 3])
    idx2name = densepose_idx_to_name()
    visible = [idx2name[i] for i in range(1, 25) if i in np.unique(dp_mask)]
    if 'Upper Leg Left' not in visible:
        grad_mask[1] = 0
    if 'Upper Leg Right' not in visible:
        grad_mask[2] = 0
    if 'Lower Leg Left' not in visible:
        grad_mask[4] = 0
    if 'Lower Leg Right' not in visible:
        grad_mask[5] = 0
    if 'Left Foot' not in visible:
        grad_mask[7] = 0
        grad_mask[10] = 0
    if 'Right Foot' not in visible:
        grad_mask[8] = 0
        grad_mask[11] = 0
    if 'Upper Arm Left' not in visible:
        grad_mask[16] = 0
    if 'Upper Arm Right' not in visible:
        grad_mask[17] = 0
    if 'Lower Arm Left' not in visible:
        grad_mask[18] = 0
    if 'Lower Arm Right' not in visible:
        grad_mask[19] = 0
    if 'Left Hand' not in visible:
        grad_mask[20] = 0
        grad_mask[22] = 0
    if 'Right Hand' not in visible:
        grad_mask[21] = 0
        grad_mask[23] = 0
    if 'Head' not in visible:
        grad_mask[12] = 0
        grad_mask[15] = 0
    return grad_mask.reshape(-1)
