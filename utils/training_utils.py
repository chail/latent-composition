import torch
import argparse
import itertools

### arguments

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--netE_type', type=str, required=True, help='type of encoder architecture; e.g. resnet-18, resnet-34')
    parser.add_argument('--netG', type=str, required=True, help="generator to load")
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--outf', default='.', help='folder to output model checkpoints')
    parser.add_argument('--seed', default=0, type=int, help='manual seed')
    parser.add_argument('--lambda_latent', default=1.0, type=float, help='loss weighting (latent recovery)')
    parser.add_argument('--lambda_mse', default=1.0, type=float, help='loss weighting (image mse)')
    parser.add_argument('--lambda_lpips', default=1.0, type=float, help='loss weighting (image perceptual)')
    parser.add_argument('--lambda_id', default=0.0, type=float, help='loss weighting (optional identity loss for faces)')
    parser.add_argument('--netE', default='', help="path to netE (to continue training)")
    parser.add_argument('--finetune', type=str, default='', help="finetune from weights at this path")
    parser.add_argument('--masked', action='store_true', help="train with masking")
    parser.add_argument('--vae_like', action='store_true', help='train with masking, predict mean and sigma (not used in paper)')
    return parser

### checkpointing

def make_checkpoint(netE, optimizer, epoch, val_loss, save_path):
    sd = {
        'state_dict': netE.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }
    torch.save(sd, save_path)

### loss functions

def cor_square_error_loss(x, y, eps=1e-8):
    # Analogous to MSE, but in terms of Pearson's correlation
    return (1.0 - torch.nn.functional.cosine_similarity(x, y, eps=eps)).mean()

### interpolation utilities

def make_ipol_layer(size):
    return torch.nn.AdaptiveAvgPool2d((size, size))
    # return InterpolationLayer(size)

class InterpolationLayer(torch.nn.Module):
    def __init__(self, size):
        super(InterpolationLayer, self).__init__()
        self.size=size

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x, size=self.size, mode='area')


### dataset utilities

def training_loader(nets, batch_size, global_seed=0):
    '''
    Returns an infinite generator that runs through randomized z
    batches, forever.
    '''
    g_epoch = 1
    while True:
        z_data = nets.sample_zs(n=10000, seed=g_epoch+global_seed,
                                device='cpu')
        dataloader = torch.utils.data.DataLoader(
                z_data,
                shuffle=False,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=True)
        for batch in dataloader:
            yield batch
        g_epoch += 1

def testing_loader(nets, batch_size, global_seed=0):
    '''
    Returns an a short iterator that returns a small set of test data.
    '''
    z_data = nets.sample_zs(n=10*batch_size, seed=global_seed,
                            device='cpu')
    dataloader = torch.utils.data.DataLoader(
            z_data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True)
    return dataloader

def epoch_grouper(loader, epoch_size, num_epochs=None):
    '''
    To use with the infinite training loader: groups the training data
    batches into epochs of the given size.
    '''
    it = iter(loader)
    epoch = 0
    while True:
        chunk_it = itertools.islice(it, epoch_size)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)
        epoch += 1
        if num_epochs is not None and epoch >= num_epochs:
            return
