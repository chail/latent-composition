from utils import renormalize, pbar, pidfile, compositions, parallelfolder
import os
import os.path as osp
import torch
import numpy as np
from PIL import Image
import argparse
from networks import networks
os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/%s_cpp/' % os.environ['USER']
from torchvision import transforms

def main(args):

    # make all the directories
    os.makedirs(osp.join(args.outdir, 'composite_original'), exist_ok=True)
    os.makedirs(osp.join(args.outdir, 'composite_mask'), exist_ok=True)
    os.makedirs(osp.join(args.outdir, 'inverted_RGBM'), exist_ok=True)
    os.makedirs(osp.join(args.outdir, 'poisson'), exist_ok=True)

    model_type = args.model
    domain = args.domain
    nets_RGBM = networks.define_nets(model_type, domain)
    compositer=compositions.get_compositer(domain)(nets_RGBM)

    if args.input_source == 'images':
        assert(args.domain in args.data_path) # sanity check!
        outdim = nets_RGBM.setting['outdim']
        transform = transforms.Compose([
                    transforms.Resize(outdim),
                    transforms.CenterCrop(outdim),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        dset = parallelfolder.ParallelImageFolders([args.data_path], transform=transform, lazy_init=False)
        print("Using dataset of length: %d" % len(dset))

    def resize2image(tensor, method=Image.ANTIALIAS):
        return (renormalize.as_image(tensor[0])
                .resize((args.im_size, args.im_size), method))

    for i in pbar(range(args.num_samples)):
        with torch.no_grad():
            if args.input_source == 'samples':
                rng = np.random.RandomState(i)
                indices = rng.choice(compositer.total_samples,
                                     len(compositer.ordered_labels))
                composite_data = compositer(indices)
            elif args.input_source == 'images':
                rng = np.random.RandomState(i)
                indices = rng.choice(len(dset),
                                     len(compositer.ordered_labels))
                images = [dset[i][0] for i in indices]
                composite_data = compositer(indices=None, imgs=images)
            resize2image(composite_data.composite_image).save(os.path.join(
                args.outdir, 'composite_original',
                'sample%06d_composite_original.png' % i))
            resize2image(composite_data.composite_mask, method=Image.NEAREST).save(
                os.path.join(args.outdir, 'composite_mask',
                'sample%06d_composite_mask.png' % i))
            resize2image(composite_data.inverted_RGBM).save(os.path.join(
                args.outdir, 'inverted_RGBM',
                'sample%06d_inverted_RGBM.png' % i))
            poisson = compositions.poisson_blend_layers(
                composite_data.parts_image, composite_data.parts_mask)
            poisson = poisson.resize((args.im_size, args.im_size), Image.ANTIALIAS)
            poisson.save(osp.join(args.outdir, 'poisson',
                'sample%06d_poisson.png' % i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Composition Experiment')
    parser.add_argument('--model', required=True,
                        help='proggan, stylegan')
    parser.add_argument('--domain', required=True,
                        help='church, ffhq... etc')
    parser.add_argument('--input_source', required=True,
                        help='samples, image')
    parser.add_argument('--outdir', required=True,
                        help='output directory')
    parser.add_argument('--data_path',
                        help='used if input_source = image')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for z samples')
    parser.add_argument('--num_samples', type=int, default=50000,
                        help='number of samples')
    parser.add_argument('--im_size', type=int, help='resize all output to this size')

    args = parser.parse_args()
    args.outdir = args.outdir.format(**vars(args))
    os.makedirs(args.outdir, exist_ok=True)
    pidfile.exit_if_job_done(args.outdir)
    main(args)
    pidfile.mark_job_done(args.outdir)
