import numpy as np
from prdc import compute_prdc
import torch
import os
from utils import renormalize, features, parallelfolder, pbar, pidfile
from torchvision import transforms
from torch.utils.data import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time
from PIL import Image


def load_or_compute_features(path, args):
    saved_features = os.path.join(path, 'metrics/prdc/features.npz')
    if os.path.isfile(saved_features):
        print("found saved features: %s" % saved_features)
        return np.load(saved_features)['features']

    transform = transforms.Compose([
        transforms.Resize(args.load_size, Image.ANTIALIAS),
        transforms.CenterCrop(args.load_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # uses imagenet normalization, with pretrained vgg features
    dset = parallelfolder.ParallelImageFolders([path], transform=transform,
                                               lazy_init=False)
    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False,
                        pin_memory=False, num_workers=args.workers)
    model = features.vgg16().eval().cuda()
    feature_list = []
    for data in pbar(loader):
        data = data[0].cuda()
        with torch.no_grad():
            feature_list.append(model(data).cpu().numpy())
    feature_list = np.concatenate(feature_list)
    os.makedirs(os.path.join(path, 'metrics/prdc'), exist_ok=True)
    np.savez(saved_features, features=feature_list)
    return feature_list

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, nargs=2,
        help='Path to the generated images or to .npz statistic files')
    parser.add_argument('--outdir', type=str, default=None,
                        help='path to save computed prdc')
    parser.add_argument('--nearest_k', type=int, default=5,
                        help='nearest k to use for prdc')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size to compute vgg features')
    parser.add_argument('--workers', type=int, default=4,
                        help='data loading workers')
    parser.add_argument('--load_size', type=int, default=256,
                        help='size to load images at')

    args = parser.parse_args()
    outdir = args.outdir
    if outdir is None:
        outdir = os.path.join(args.path[-1], 'metrics/prdc')
    pidfile.exit_if_job_done(outdir, redo=False)

    real_features = load_or_compute_features(args.path[0], args)
    fake_features = load_or_compute_features(args.path[1], args)
    start_time = time.time()
    metrics = compute_prdc(real_features=real_features,
                           fake_features=fake_features,
                           nearest_k=args.nearest_k)
    end_time = time.time()
    print("Computed PRDC in %s min:" % ((end_time-start_time)/60))
    for k,v in metrics.items():
        print("{}: {}".format(k, v))
    np.savez(os.path.join(outdir, 'prdc.npz'), **metrics)

    pidfile.mark_job_done(outdir)



