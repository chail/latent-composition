import numpy as np
import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import parallelfolder, pbar, pidfile, losses
from PIL import Image

def compute_distances(args):
    transform = transforms.Compose([
                    transforms.Resize(args.load_size, Image.ANTIALIAS),
                    transforms.CenterCrop(args.load_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    rec_path = args.path[0].rstrip('/')
    if 'car' in rec_path:
        assert(args.crop_aspect_car) # sanity check
    gt_path = os.path.join(os.path.dirname(rec_path),'composite_original')
    mask_path = os.path.join(os.path.dirname(rec_path),'composite_mask')
    dset_GT = parallelfolder.ParallelImageFolders([gt_path],
                                                  transform=transform,
                                                  lazy_init=False,
                                                  return_path=True)
    dset_mask = parallelfolder.ParallelImageFolders([mask_path],
                                                  transform=transform,
                                                  lazy_init=False,
                                                  return_path=True)
    dset_rec = parallelfolder.ParallelImageFolders([rec_path],
                                                  transform=transform,
                                                  lazy_init=False,
                                                  return_path=True)
    gt_loader = DataLoader(dset_GT, batch_size=args.batch_size, shuffle=False,
                           pin_memory=False, num_workers=2)

    mask_loader = DataLoader(dset_mask, batch_size=args.batch_size, shuffle=False,
                           pin_memory=False, num_workers=2)
    rec_loader = DataLoader(dset_rec, batch_size=args.batch_size, shuffle=False,
                           pin_memory=False, num_workers=2)

    l1_distance = losses.Masked_L1_Loss()
    lpips_alex = losses.Masked_LPIPS_Loss(net='alex')
    lpips_vgg = losses.Masked_LPIPS_Loss(net='vgg')
    loss_metrics = dict(l1=[], lpips_alex=[], lpips_vgg=[])
    for (gt, mask, rec) in zip(pbar(gt_loader), mask_loader, rec_loader):
        # sanity check the paths for consistency
        for p1, p2, p3 in zip(gt[1][0], mask[1][0], rec[1][0]):
            sample = p1.split('/')[-1].split('_')[0]
            assert(p2.split('/')[-1].split('_')[0] == sample)
            assert(p3.split('/')[-1].split('_')[0] == sample)
        with torch.no_grad():
            gt_tensor = gt[0][0].cuda()
            rec_tensor = rec[0][0].cuda()
            mask_tensor = mask[0][0] # 0 to 1
            mask_tensor[mask_tensor < 0.5] = 0
            assert(torch.max(mask_tensor) == 1)
            assert(torch.min(mask_tensor) == 0)
            if args.crop_aspect_car:
                # don't compute similarity on black padding
                aspect_border = int(0.125 * args.load_size)
                mask_tensor[:, :, :aspect_border, :] = 0
                mask_tensor[:, :, -aspect_border:, :] = 0
            mask_tensor = mask_tensor[:, :1, :, :].cuda()
            batch_l1 = l1_distance(rec_tensor, gt_tensor,
                                   mask_tensor).cpu().numpy().squeeze()
            batch_lpips_alex = lpips_alex(rec_tensor, gt_tensor,
                                   mask_tensor).cpu().numpy().squeeze()
            batch_lpips_vgg = lpips_vgg(rec_tensor, gt_tensor,
                                   mask_tensor).cpu().numpy().squeeze()
            loss_metrics['l1'].append(batch_l1)
            loss_metrics['lpips_alex'].append(batch_lpips_alex)
            loss_metrics['lpips_vgg'].append(batch_lpips_vgg)

    loss_metrics['l1'] = np.concatenate(loss_metrics['l1'])
    loss_metrics['lpips_alex'] = np.concatenate(loss_metrics['lpips_alex'])
    loss_metrics['lpips_vgg'] = np.concatenate(loss_metrics['lpips_vgg'])
    loss_metrics['l1_avg'] = np.mean(loss_metrics['l1'])
    loss_metrics['lpips_alex_avg'] = np.mean(loss_metrics['lpips_alex'])
    loss_metrics['lpips_vgg_avg'] = np.mean(loss_metrics['lpips_vgg'])
    return loss_metrics

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, nargs=1,
                        help='Path to the generated images')
    parser.add_argument('--outdir', type=str, default=None,
                        help='path to save computed prdc')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size to compute vgg features')
    parser.add_argument('--workers', type=int, default=4,
                        help='data loading workers')
    parser.add_argument('--load_size', type=int, default=256,
                        help='size to load images at')
    parser.add_argument('--crop_aspect_car', action='store_true',
                        help='crop out border padding for cars')

    args = parser.parse_args()
    outdir = args.outdir
    if outdir is None:
        outdir = os.path.join(args.path[-1], 'metrics/distances')
    pidfile.exit_if_job_done(outdir, redo=False)

    metrics = compute_distances(args)
    for k,v in metrics.items():
        if 'avg' in k:
            print("{}: {}".format(k, v))
    np.savez(os.path.join(outdir, 'distances.npz'), **metrics)

    pidfile.mark_job_done(outdir)

