import torch
import numpy as np
from torch.nn.functional import interpolate

def mask_upsample(im_tensor, mask_cent=0.5, threshold=None):
    batch_size = im_tensor.shape[0]
    image_size = im_tensor.shape[-1]
    m = torch.rand(batch_size,1,6,6).to(im_tensor.device)
    m = interpolate(m, size=image_size, mode='bilinear',
                    align_corners=False)
    if threshold is not None:
        cutoff = threshold
    else:
        cutoff = torch.rand((1,))*0.7+0.3
        cutoff = cutoff[0] # 0.3 to 1
    mask = (m < cutoff).float()
    return im_tensor * mask, mask - mask_cent

def mask_patches(im_tensor, p=0.1, sizes=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], mask_cent=0.5):
    N, C, H, W = im_tensor.shape
    mask = torch.zeros((N, 1, H, W)).to(im_tensor.device)
    hints = torch.zeros_like(im_tensor)
    for nn in range(N):
        pp = 0
        cont_cond = True
        while(cont_cond):
            # geometric distribution
            cont_cond = np.random.rand() < (1-p)
            if (not cont_cond and pp > 0):
                continue
            pp += 1
            P = np.random.choice(sizes, 2)

            # random uniform sampling
            PH = int(P[0]*H)
            PW = int(P[1]*W)
            h = np.random.randint(H-PH+1)
            w = np.random.randint(W-PW+1)

            hints[nn, :, h:h+PH, w:w+PW] = im_tensor[nn, :, h:h+PH, w:w+PW]
            mask[nn, :, h:h+PH, w:w+PW] = 1
    return hints, mask - mask_cent

