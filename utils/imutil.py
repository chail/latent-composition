from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import torch
import numpy as np
from . import renormalize

# Arrange list of images in a grid with padding
# adapted from: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb
def imgrid(imarray_np, cols=5, pad=1):
    if imarray_np.dtype != np.uint8:
        raise ValueError('imgrid input imarray_np must be uint8')
    if imarray_np.shape[1] in [1, 3]:
        # reorder channel dimension
        imarray_np = np.transpose(imarray_np, (0, 2, 3, 1))
    pad = int(pad)
    assert pad >= 0
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray_np.shape
    rows = int(np.ceil(N / float(cols)))
    batch_pad = rows * cols - N
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad, 0]
    pad_arg = [[0, p] for p in post_pad]
    imarray_np = np.pad(imarray_np, pad_arg, 'constant', constant_values=255)
    H += pad
    W += pad
    grid = (imarray_np
            .reshape(rows, cols, H, W, C)
            .transpose(0, 2, 1, 3, 4)
            .reshape(rows*H, cols*W, C))
    if pad:
        grid = grid[:-pad, :-pad]
    return grid

def draw_contour(image, mask, color=[225, 225, 0]):
    assert(np.ndim(image) == 3)
    image_copy = np.copy(image)
    if not isinstance(mask, list):
        mask = [mask]
    for m in mask:
        assert(np.ndim(m) == 2)
        border = border_from_mask(m)
        contour_y, contour_x = np.where(border)
        image_copy[contour_y, contour_x, :] = color
    return image_copy


def border_from_mask(a):
    out = np.zeros_like(a)
    h = (a[:-1,:] != a[1:,:])
    v = (a[:,:-1] != a[:,1:])
    d = (a[:-1,:-1] != a[1:,1:])
    u = (a[1:,:-1] != a[:-1,1:])
    out[:-1,:-1] |= d
    out[1:,1:] |= d
    out[1:,:-1] |= u
    out[:-1,1:] |= u
    out[:-1,:] |= h
    out[1:,:] |= h
    out[:,:-1] |= v
    out[:,1:] |= v
    out &= ~a
    return out

def draw_masked_image(image_tensor, mask_tensor, path=None, filename=None, size=256, weight=2):
    # image tensor: NCHW . Takes the first one
    # mask_tensor: NCHW . Takes the first one
    # does resizing first so that the border does not get aliased
    image_tensor_renorm = ((image_tensor[0] + 1) / 2).permute(1,2,0).detach().cpu().numpy()
    image_pil = Image.fromarray(np.uint8(image_tensor_renorm * 255)).resize((size, size), Image.ANTIALIAS)
    image_blur = image_pil.filter(ImageFilter.GaussianBlur(radius=5))
    image_np = np.array(image_pil)
    blur_np = np.array(image_blur)
    
    # make mask btw 0 and 1, takes first channel
    if torch.min(mask_tensor) < 0:
        mask_tensor += torch.min(mask_tensor)
    mask_tensor = (mask_tensor / torch.max(mask_tensor)).detach().cpu().numpy()
    # takes first channel and resizes it
    mask_pil = Image.fromarray(mask_tensor[0, 0, :, :]).resize((size, size), Image.NEAREST)
    mask_2d = np.array(mask_pil)
    border = border_from_mask(mask_2d!=0)

    
    # 1) mix image with white only
    image_white = image_np *mask_2d[...,None] + np.ones_like(image_np)* 255 * (1-mask_2d[...,None])
    image_white[border] = 0
    image_white = Image.fromarray(np.uint8(image_white))
    
    # 2) mix image with blurred whitened mask
    whitened_image_blur = (blur_np.astype('int64') + weight*255) / (weight+1)
    mixed_image_blur = image_np *mask_2d[...,None] + (whitened_image_blur) * (1-mask_2d[...,None])
    mixed_image_blur[border] = 0
    mixed_image_blur = Image.fromarray(np.uint8(np.clip(mixed_image_blur, 0, 255)))
    
    # 3) mix image with whitened mask
    whitened_image = (image_np.astype('int64') + weight*255) / (weight+1)
    mixed_image_white = image_np * mask_2d[...,None] + (whitened_image) * (1-mask_2d[...,None])
    mixed_image_white[border] = 0
    mixed_image_white = Image.fromarray(np.uint8(np.clip(mixed_image_white, 0, 255)))
    
    # save images
    if filename is not None and path is not None:
        image_white.save(os.path.join(path, filename + '_white.png'))
        mixed_image_blur.save(os.path.join(path, filename + '_mix_blur.png'))
        mixed_image_white.save(os.path.join(path, filename + '_mix_white.png'))

    # return 3 PIL images
    return image_white, mixed_image_blur, mixed_image_white


def save_image(image, path, filename, size=256):
    if isinstance(image, torch.Tensor):
        # image_tensor NCHW, takes first one
        im_pil = renormalize.as_image(image[0])
    else:
        im_pil = image
    im_pil = im_pil.resize((size, size), Image.ANTIALIAS)
    im_pil.save(os.path.join(path, filename + '.png'))
    return im_pil


def add_banner(pil_img, text, banner_height=30):
    fnt = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 25, encoding="unic")
    w, h = pil_img.size
    new_im = Image.new('RGB', (w, h+banner_height), color=(255,255,255))
    new_im.paste(pil_img, (0, banner_height))
    img_draw = ImageDraw.Draw(new_im)
    textsize = img_draw.textsize(text, font=fnt)
    limitsize = img_draw.textsize('Ay', font=fnt)
    # print(textsize)
    assert(textsize[0] < w)
    assert(limitsize[1] < banner_height)
    textX = (w - textsize[0]) // 2
    textY = (banner_height - limitsize[1]) // 2
    # print(textX)
    # print(textY)
    img_draw.text((textX, textY), text, font=fnt, fill='black')
    return new_im

