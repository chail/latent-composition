from utils import renormalize, show, pbar, pidfile
import os
import torch
import numpy as np
from PIL import Image
import argparse
from networks import networks
os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/%s_cpp/' % os.environ['USER']

def main(args):

    domain = args.domain
    batch_size = args.batch_size
    nets = networks.define_nets(args.model, args.domain, load_encoder=True,
                                device='cuda')
    zs = nets.sample_zs(args.num_samples, args.seed)

    for batch_start in pbar(range(0, args.num_samples, batch_size)):
        s = slice(batch_start, min(batch_start+batch_size, args.num_samples))
        zs_batch = zs[s]
        with torch.no_grad():
            ims_initial = nets.zs2image(zs_batch)
            ims = nets.invert(ims_initial, mask=None)
            for i, im in enumerate(ims):
                filename = os.path.join(args.outdir, 'seed%03d_sample%05d' %
                                        (args.seed, i+batch_start))
                pil_image = renormalize.as_image(im)
                if args.im_size:
                    pil_image = pil_image.resize((args.im_size, args.im_size),
                                                 Image.ANTIALIAS)
                pil_image.save(filename + '.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate GAN samples')
    parser.add_argument('--model', required=True,
                        help='proggan, stylegan')
    parser.add_argument('--domain', required=True,
                        help='church, ffhq... etc')
    parser.add_argument('--outdir', required=True,
                        help='output directory')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for z samples')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='number of samples')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--im_size', type=int, help='resize to this size')
    args = parser.parse_args()
    args.outdir = args.outdir.format(**vars(args))
    os.makedirs(args.outdir, exist_ok=True)
    pidfile.exit_if_job_done(args.outdir)
    main(args)
    cmd = f'cp utils/lightbox.html {args.outdir}/+lightbox.html'
    print(cmd)
    os.system(cmd)
    pidfile.mark_job_done(args.outdir)
