import torch
import torch.nn.functional as F
import numpy as np
import os
import os.path as osp
import sys
from collections import namedtuple
import cv2
from . import renormalize
from PIL import Image

Composite = namedtuple('Composite',
                       ['parts_image', 'parts_mask', 'composite_image',
                        'composite_filled', 'composite_mask',
                        'inverted_RGB', 'inverted_RGB_filled',
                        'inverted_RGBM', 'inverted_VAE'])

class FaceSegmenter(object):
    def __init__(self):
        sys.path.append('resources/face_parsing_pytorch/')
        from model import BiSeNet
        n_classes = 19
        seg_net = BiSeNet(n_classes=n_classes).cuda()
        save_pth = 'resources/face_parsing_pytorch/res/cp/79999_iter.pth'
        seg_net.load_state_dict(torch.load(save_pth))
        seg_net.eval()
        self.seg_net = seg_net

    def __call__(self, image):
        # image = 1CHW tensor
        assert(image.shape[0] == 1)
        im_size = image.shape[-1]
        if im_size != 512:
            with torch.no_grad():
                image = F.interpolate(image, (512, 512), mode='bilinear',
                                      align_corners=False)
        assert(image.shape[-1] == 512)
        with torch.no_grad():
            image = renormalize.as_tensor(image, source='zc',
                                          target='imagenet')
            seg = self.seg_net(image)[0]
            seg = F.interpolate(seg, (im_size, im_size), mode='bilinear',
                                align_corners=False)
            seg = torch.argmax(seg, axis=1)
        return seg


class UnifiedSegmenter(object):
    def __init__(self):
        sys.path.append('resources/ganseeing')
        from seeing import segmenter, imgviz
        upp = segmenter.UnifiedParsingSegmenter()
        self.upp = upp

    def __call__(self, image):
        # image = 1CHW tensor
        assert(image.shape[0] == 1)
        im_size = image.shape[-1]
        with torch.no_grad():
            orig_seg = self.upp.segment_batch(image)[0, 0:1]
        return orig_seg


class Compositer(object):
    def __init__(self, nets_RGBM, nets_RGB=None, nets_VAE=None,
                 total_samples=50000, seed=0, return_segmentations=False):
        self.nets_RGBM = nets_RGBM
        self.nets_RGB = nets_RGB
        self.nets_VAE = nets_VAE
        self.zs = nets_RGBM.sample_zs(total_samples, seed)
        self.total_samples = total_samples
        self.outsize = nets_RGBM.setting['outdim']
        self.return_segmentations = return_segmentations

        # subclasses should define these
        self.segmenter = None
        self.macro_clusters = None
        self.ordered_labels = None

    def __call__(self, indices=None, imgs=None):
        assert(not(indices is None and imgs is None))
        if indices is not None:
            # indices into the zs dataset
            assert(len(indices) == len(self.ordered_labels))
            assert(max(indices) < self.total_samples)
            images = [(l, i) for l, i in zip(self.ordered_labels, indices)]
        if imgs is not None:
            # provide real image input 
            assert(len(imgs) == len(self.ordered_labels))
            assert(imgs[0].shape[-1] == self.outsize)
            images = [(l, i) for l, i in zip(self.ordered_labels, imgs)]

        composite = (torch.Tensor([0, 0, 0]).view(3, 1, 1)
                     .repeat(1, self.outsize, self.outsize)[None]
                     .float().cuda())
        mask_composite = torch.zeros_like(composite)[:, [0], :, :] # 1 channel

        composite_image_output = []
        composite_mask_output = []
        segmentations = []
        for i, (label, sample) in enumerate(images):
            with torch.no_grad():
                if indices is not None:
                    z = self.zs[sample][None]
                    im = self.nets_RGBM.zs2image(z)
                else:
                    im = sample[None].cuda()

                seg = self.segmenter(im)
                segmentations.append(seg.cpu())

            # masked
            assert(seg.shape[0] == 1)
            mask = torch.zeros_like(im)
            for l in self.macro_clusters[label]:
                _,h,w = torch.where(seg == l)
                mask[:, :, h, w] = 1
                mask_composite[:, :, h, w] = 1
            masked_im = im * mask

            composite_image_output.append(im)
            composite_mask_output.append(mask)
            # fill in background
            if i == 0:
                background_color = (torch.sum(im * mask, axis=(-1, -2)) / torch.sum(mask, axis=(-2, -1))).view(1, 3, 1, 1)
                if torch.any(torch.isnan(background_color)):
                    # if the background part is not detected, just take
                    # the mean of the image
                    background_color = (torch.mean(im, axis=(-1, -2))
                                        .view(1, 3, 1, 1))
                background = background_color.repeat(1, 1, self.outsize, self.outsize)
                composite_filled = background

            # add to composite
            composite = im * mask + composite * (1-mask)
            composite_filled = im * mask + composite_filled * (1-mask)

        with torch.no_grad():
            regenerated_RGBM = self.nets_RGBM.invert(composite, mask_composite)
            if self.nets_RGB is not None:
                regenerated_RGB = self.nets_RGB.invert(composite, mask=None)
                regenerated_RGB_filled = self.nets_RGB.invert(
                    composite_filled, mask=None)
            else:
                regenerated_RGB = None
                regenerated_RGB_filled = None

            if self.nets_VAE is not None:
                regenerated_VAE = self.nets_VAE.invert(composite,
                                                       mask_composite)
            else:
                regenerated_VAE = None

        composite_data = Composite(torch.cat(composite_image_output),
                         torch.cat(composite_mask_output),
                         composite, composite_filled, mask_composite,
                         regenerated_RGB, regenerated_RGB_filled,
                         regenerated_RGBM, regenerated_VAE)

        if self.return_segmentations:
            return (composite_data, segmentations)
        return composite_data


class FaceCompositer(Compositer):
    def __init__(self, *args, **kwargs):
        super(FaceCompositer, self).__init__(*args, **kwargs)
        self.macro_clusters = {
            'background': [0, 16], # background, clothes
            'skin': [1, 14, 15, 7, 8, 9],
            'eye': [2, 3, 4, 5, 6],
            'nose': [10],
            'mouth': [11, 12, 13],
            'hair': [17]
        }
        self.ordered_labels = ['background', 'skin', 'eye', 'mouth', 'nose', 'hair']
        self.segmenter = FaceSegmenter()


class ChurchCompositer(Compositer):
    def __init__(self, *args, **kwargs):
        super(ChurchCompositer, self).__init__(*args, **kwargs)
        self.macro_clusters = {
            'building': [5, 136], # building, skyscraper
            'sky': [2],
            'tree': [4],
            'foreground': [11, 10, 17, 78, 14], # grass road sidewalk plant path
        }
        self.ordered_labels = ['sky', 'building', 'tree', 'foreground']
        self.segmenter = UnifiedSegmenter()


class LivingroomCompositer(Compositer):
    def __init__(self, *args, **kwargs):
        super(LivingroomCompositer, self).__init__(*args, **kwargs)
        self.macro_clusters = {
            'window': [9, 23], # window + curtain
            'floor': [3, 38], # floor + carpet
            'wall': [1],
            'ceiling': [7, 18, 75], # ceiling + light + chandelier
            'sofa': [29, 12, 34, 48, 44], # sofa + chair + cushion + pillow + armchair
            'coffee table': [59, 131], # coffee table + ottoman
            'painting': [15],
            'fireplace': [92, 19], # fireplace + cabinet
        }
        self.ordered_labels = ['floor', 'ceiling', 'wall', 'painting', 'window', 'fireplace', 'sofa', 'coffee table']
        self.segmenter = UnifiedSegmenter()


class CarCompositer(Compositer):
    def __init__(self, *args, **kwargs):
        super(CarCompositer, self).__init__(*args, **kwargs)
        self.macro_clusters = {
            'building': [5, 1], # building, wall
            'sky': [2],
            'tree': [4, 11], # tree, grass
            'foreground': [22, 31, 10], # ground / earth / road
            'car': [13]
        }
        self.ordered_labels = ['sky', 'building', 'tree', 'foreground', 'car']
        self.segmenter = UnifiedSegmenter()

class SaliencyCompositer(object):
    def __init__(self, nets_RGBM, nets_RGB=None, nets_VAE=None,
                 total_samples=50000, seed=0):
        self.nets_RGBM = nets_RGBM
        self.nets_RGB = nets_RGB
        self.nets_VAE = nets_VAE
        self.zs = nets_RGBM.sample_zs(total_samples, seed)
        self.total_samples = total_samples
        self.outsize = nets_RGBM.setting['outdim']

        sys.path.append('resources/PiCANet-Implementation')
        from network import Unet
        from dataset import CustomDataset
        ckpt = 'resources/PiCANet-Implementation/36epo_383000step.ckpt'
        state_dict = torch.load(ckpt)
        model = Unet().cuda()
        model.load_state_dict(state_dict)
        self.model = model

    def __call__(self, src_sample, tgt_sample):
        z_src = self.zs[src_sample]
        z_tgt = self.zs[tgt_sample]
        with torch.no_grad():
            img_src = self.nets_RGBM.zs2image(z_src[None])
            img_tgt = self.nets_RGBM.zs2image(z_tgt[None])
            img_src_224= F.interpolate(img_src, 224, mode='bilinear', align_corners=False)
            img_src_224 = renormalize.as_tensor(img_src_224, source='zc', target='pt')
            img_tgt_224 = F.interpolate(img_tgt, 224, mode='bilinear', align_corners=False)
            img_tgt_224 = renormalize.as_tensor(img_tgt_224, source='zc', target='pt')
            outdim = self.nets_RGBM.setting['outdim']
            pred_src = F.interpolate(self.model(img_src_224)[0][5].data, outdim, mode='bilinear', align_corners=False)
            pred_tgt = F.interpolate(self.model(img_tgt_224)[0][5].data, outdim, mode='bilinear', align_corners=False)
            mask = (pred_src > 0.5).float()
            composite = mask * img_src + (1-mask) * img_tgt
            # note: we don't need a mask here, it should just invert
            # the full composite image
            regenerated = self.nets_RGBM.invert(composite) # , mask)

        return Composite(torch.cat([img_src, img_tgt]),
                         torch.cat([mask[:, [0], :, :],
                                    torch.ones_like(img_src)[:, [0], :, :]]),
                         composite, None,  mask[:, [0], :, :],
                         regenerated, None, None, None)


def poisson_blend_layers(composite_parts_image, composite_parts_mask):
    targets = [cv2.cvtColor(np.array(renormalize.as_image(part)), cv2.COLOR_RGB2BGR)
               for part in composite_parts_image]
    masks = [mask.transpose(1,2,0) for mask in composite_parts_mask.cpu().numpy()]

    blended_result = targets[0]
    for i in range(1, len(targets)):
        h, w, c = targets[i].shape
        if np.mean(masks[i]) < 0.1:
            continue
        blended_result = cv2.seamlessClone(
            targets[i],
            blended_result,
            (255 * masks[i]).astype(np.uint8),
            (w//2, h//2),
            cv2.NORMAL_CLONE
        )
    return Image.fromarray(cv2.cvtColor(blended_result, cv2.COLOR_BGR2RGB))

def get_compositer(domain):
    if 'ffhq' in domain or 'celebahq' in domain:
        return FaceCompositer
    if 'church' in domain:
        return ChurchCompositer
    if 'livingroom' in domain:
        return LivingroomCompositer
    if domain == 'car':
        return CarCompositer
    # could also add some other domains, e.g. bedroom
    # if we train the encoders
    return None

