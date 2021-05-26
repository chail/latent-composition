from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import oyaml as yaml
from utils import pbar, util, masking, losses, training_utils
from networks import networks, stylegan_networks
os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/%s_cpp/' % os.environ['USER']

from networks.psp import id_loss

def train(opt):
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if has_cuda else "cpu")
    batch_size = int(opt.batchSize)

    cudnn.benchmark = True

    # tensorboard
    writer = SummaryWriter(log_dir='training/runs/%s' % os.path.basename(opt.outf))

    # load the generator
    nets = networks.define_nets('stylegan', opt.netG, use_RGBM=opt.masked,
                                use_VAE=opt.vae_like, ckpt_path=None,
                                load_encoder=False, device=device)
    netG = nets.generator
    util.set_requires_grad(False, netG)
    netG.eval()

    # find output shape
    outdim = nets.setting['outdim']
    nz = nets.setting['nlatent'] # encodes to W+ space, dim=z_dim*n_layers
    print(outdim)

    # get the encoder
    depth = int(opt.netE_type.split('-')[-1])
    has_masked_input = opt.masked or opt.vae_like
    assert(not (opt.masked and opt.vae_like)), "specify 1 of masked or vae_like"
    netE = stylegan_networks.load_stylegan_encoder(domain=None, nz=nz,
                                                   outdim=outdim,
                                                   use_RGBM=opt.masked,
                                                   use_VAE=opt.vae_like,
                                                   resnet_depth=depth,
                                                   ckpt_path=None)
    netE = netE.to(device).train()
    nets.encoder = netE

    # losses + optimizers
    mse_loss = nn.MSELoss()
    # l1_loss = nn.L1Loss()
    perceptual_loss = losses.LPIPS_Loss(net='vgg', use_gpu=has_cuda)
    util.set_requires_grad(False, perceptual_loss)
    if opt.netG not in ['ffhq', 'celebahq']:
        # only uses identity loss for face domains
        assert(opt.lambda_id == 0.)
    if opt.lambda_id > 0:
        identity_loss = id_loss.IDLoss().cuda().eval()
        util.set_requires_grad(False, identity_loss)
    # resize img to 256 before lpips computation
    reshape = training_utils.make_ipol_layer(256)
    optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    start_ep = 0
    best_val_loss = float('inf')

    # latent datasets
    train_loader = training_utils.training_loader(nets, batch_size, opt.seed)
    test_loader = training_utils.testing_loader(nets, batch_size, opt.seed)

    # load data from checkpoint
    assert(not (opt.netE and opt.finetune)), "specify 1 of netE or finetune"
    if opt.finetune:
        checkpoint = torch.load(opt.finetune)
        sd = checkpoint['state_dict']
        netE.load_state_dict(sd)
    if opt.netE:
        checkpoint = torch.load(opt.netE)
        netE.load_state_dict(checkpoint['state_dict'])
        optimizerE.load_state_dict(checkpoint['optimizer'])
        start_ep = checkpoint['epoch'] + 1
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']

    # uses 1600 samples per epoch, computes number of batches
    # based on batch size
    epoch_batches = 1600 // batch_size
    for epoch, epoch_loader in enumerate(pbar(
        training_utils.epoch_grouper(train_loader, epoch_batches),
        total=(opt.niter-start_ep)), start_ep):

        # stopping condition
        if epoch > opt.niter:
            break

        # run a train epoch of epoch_batches batches
        for step, z_batch in enumerate(pbar(
            epoch_loader, total=epoch_batches), 1):
            z_batch = z_batch.to(device)
            netE.zero_grad()

            with torch.no_grad():
                fake_ws = nets.generator.gen.style(z_batch)
                fake_im = netG(fake_ws)

            if has_masked_input:
                hints_fake, mask_fake = masking.mask_upsample(fake_im)
                mask_fake = mask_fake + 0.5 # trained in range [0, 1]
                encoded = netE(torch.cat([hints_fake, mask_fake], dim=1))
                # encoded.shape = [batch, layers, nets.setting['nz']]
                if opt.masked:
                    regenerated = netG(encoded)
                elif opt.vae_like:
                    # determine number of stylegan layers
                    nlayers = nets.setting['nlatent'] // nets.setting['nz']
                    sample = torch.randn_like(encoded[:, nlayers:, :])
                    encoded_mean  = encoded[:, nlayers:, :]
                    encoded_sigma = torch.exp(encoded[:, :nlayers, :])
                    reparam = encoded_mean + encoded_sigma * sample
                    regenerated = netG(reparam)
                    encoded = encoded_mean # just use mean in z loss
            else:
                # standard RGB encoding
                encoded = netE(fake_im)
                regenerated = netG(encoded)

            # compute loss
            fake_wplus = torch.stack([fake_ws] * encoded.shape[1], dim=1)
            loss_latent = mse_loss(encoded, fake_wplus)
            loss_mse = mse_loss(regenerated, fake_im)
            loss_perceptual = perceptual_loss.forward(
                reshape(regenerated), reshape(fake_im)).mean()
            if opt.lambda_id > 0:
                loss_id, sim_improvement, id_logs = identity_loss(
                    reshape(regenerated), reshape(fake_im), reshape(fake_im))
            else:
                loss_id = torch.Tensor((0.,)).to(device)
            loss = (opt.lambda_latent * loss_latent
                    + opt.lambda_mse * loss_mse
                    + opt.lambda_lpips * loss_perceptual
                    + opt.lambda_id * loss_id)

            # optimize
            loss.backward()
            optimizerE.step()

            # send losses to tensorboard
            if step % 20 == 0:
                total_batches = epoch * epoch_batches + step
                writer.add_scalar('loss/train_latent', loss_latent, total_batches)
                writer.add_scalar('loss/train_mse', loss_mse, total_batches)
                writer.add_scalar('loss/train_id', loss_id, total_batches)
                writer.add_scalar('loss/train_lpips', loss_perceptual,
                                  total_batches)
                writer.add_scalar('loss/train_total', loss, total_batches)
                pbar.print("Epoch %d step %d Losses z %0.4f mse %0.4f id %0.4f lpips %0.4f total %0.4f"
                           % (epoch, step, loss_latent.item(), loss_mse.item(),
                              loss_id.item(), loss_perceptual.item(), loss.item()))
            if step == 1:
                total_batches = epoch * epoch_batches + step
                if has_masked_input:
                    grid = vutils.make_grid(
                        torch.cat((reshape(fake_im), reshape(hints_fake),
                                   reshape(regenerated))),
                        nrow=8, normalize=True, scale_each=(-1, 1))
                else:
                    grid = vutils.make_grid(
                        torch.cat((reshape(fake_im), reshape(regenerated))), nrow=8,
                        normalize=True, scale_each=(-1, 1))
                writer.add_image('Train Image', grid, total_batches)

        # updated to run a small set of test zs 
        # rather than a single fixed batch
        netE.eval()
        test_metrics = {
            'loss_latent': util.AverageMeter('loss_latent'),
            'loss_mse': util.AverageMeter('loss_mse'),
            'loss_perceptual': util.AverageMeter('loss_perceptual'),
            'loss_id': util.AverageMeter('loss_id'),
            'loss_total': util.AverageMeter('loss_total'),
        }
        for step, test_zs in enumerate(pbar(test_loader), 1):
            with torch.no_grad():
                fake_ws = nets.generator.gen.style(test_zs.to(device))
                fake_im = netG(fake_ws)

                if has_masked_input:
                    hints_fake, mask_fake = masking.mask_upsample(fake_im)
                    mask_fake = mask_fake + 0.5 # trained in range [0, 1]
                    encoded = netE(torch.cat([hints_fake, mask_fake], dim=1))
                    if opt.masked:
                        regenerated = netG(encoded)
                    elif opt.vae_like:
                        nlayers = nets.setting['nlatent'] // nets.setting['nz']
                        sample = torch.randn_like(encoded[:, nlayers:, :])
                        encoded_mean  = encoded[:, nlayers:, :]
                        encoded_sigma = torch.exp(encoded[:, :nlayers, :])
                        reparam = encoded_mean + encoded_sigma * sample
                        regenerated = netG(reparam)
                        encoded = encoded_mean # just use mean in z loss
                else:
                    # standard RGB encoding
                    encoded = netE(fake_im)
                    regenerated = netG(encoded)

                # compute loss
                fake_wplus = torch.stack([fake_ws] *encoded.shape[1], dim=1)
                loss_latent = mse_loss(encoded, fake_wplus)
                loss_mse = mse_loss(regenerated, fake_im)
                loss_perceptual = perceptual_loss.forward(
                    reshape(regenerated), reshape(fake_im)).mean()
                if opt.lambda_id > 0:
                    loss_id, sim_improvement, id_logs = identity_loss(
                        reshape(regenerated), reshape(fake_im), reshape(fake_im))
                else:
                    loss_id = torch.Tensor((0.,)).to(device)
                loss = (opt.lambda_latent * loss_latent
                        + opt.lambda_mse * loss_mse
                        + opt.lambda_lpips * loss_perceptual
                        + opt.lambda_id * loss_id)

            # update running avg
            test_metrics['loss_latent'].update(loss_latent)
            test_metrics['loss_mse'].update(loss_mse)
            test_metrics['loss_id'].update(loss_id)
            test_metrics['loss_perceptual'].update(loss_perceptual)
            test_metrics['loss_total'].update(loss)

            # save a fixed batch for visualization
            if step == 1:
                if has_masked_input:
                    grid = vutils.make_grid(
                        torch.cat((reshape(fake_im), reshape(hints_fake),
                                   reshape(regenerated))),
                        nrow=8, normalize=True, scale_each=(-1, 1))
                else:
                    grid = vutils.make_grid(
                        torch.cat((reshape(fake_im), reshape(regenerated))), nrow=8,
                        normalize=True, scale_each=(-1, 1))

        # send to tensorboard
        writer.add_scalar('loss/test_latent', loss_latent, epoch)
        writer.add_scalar('loss/test_mse', loss_mse, epoch)
        writer.add_scalar('loss/test_id', loss_id, epoch)
        writer.add_scalar('loss/test_lpips', loss_perceptual, epoch)
        writer.add_scalar('loss/test_total', loss, epoch)
        writer.add_image('Test Image', grid, epoch)
        netE.train()

        # do checkpointing
        if epoch % 500 == 0 or epoch == opt.niter:
            training_utils.make_checkpoint(
                netE, optimizerE, epoch,
                test_metrics['loss_total'].avg.item(),
                '%s/netE_epoch_%d.pth' % (opt.outf, epoch))
        # if epoch == opt.niter:
        #     cmd = 'ln -s netE_epoch_%d.pth %s/model_final.pth' % (epoch, opt.outf)
        #     os.system(cmd)
        if test_metrics['loss_total'].avg.item() < best_val_loss:
            # modified to save based on test zs loss rather than
            # final model at the end
            pbar.print("Checkpointing at epoch %d" % epoch)
            training_utils.make_checkpoint(
                netE, optimizerE, epoch,
                test_metrics['loss_total'].avg.item(),
                '%s/netE_epoch_best.pth' % (opt.outf))
            best_val_loss = test_metrics['loss_total'].avg.item()


if __name__ == '__main__':
    parser = training_utils.make_parser()
    opt = parser.parse_args()
    print(opt)

    opt.outf = opt.outf.format(**vars(opt))

    os.makedirs(opt.outf, exist_ok=True)
    # save options
    with open(os.path.join(opt.outf, 'optE.yml'), 'w') as f:
        yaml.dump(vars(opt), f, default_flow_style=False)

    train(opt)
