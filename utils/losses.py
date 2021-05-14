import torch
import torch.nn as nn
import lpips

class LPIPS_Loss(nn.Module):
    def __init__(self, model='net-lin', net='vgg', use_gpu=True, spatial=False):
        super(LPIPS_Loss, self).__init__()
        self.model = lpips.LPIPS(net=net, spatial=spatial).eval()
        if use_gpu:
            self.model = self.model.cuda()

    def forward(self, pred, ref):
        dist = self.model.forward(pred, ref)
        return dist

def check_loss_input(im0, im1, w):
    """ im0 is out and im1 is target and w is mask"""
    assert list(im0.size())[2:] == list(im1.size())[2:], 'spatial dim mismatch'
    if w is not None:
        assert list(im0.size())[2:] == list(w.size())[2:], 'spatial dim mismatch'

    if im1.size(0) != 1:
        assert im0.size(0) == im1.size(0)

    if w is not None and w.size(0) != 1:
        assert im0.size(0) == w.size(0)
    return

# masked lpips
# adapted from https://github.com/minyoungg/pix2latent/blob/master/pix2latent/loss_functions.py
class Masked_LPIPS_Loss(nn.Module):
    def __init__(self, net='vgg', use_gpu=True, precision='float'):
        """ LPIPS loss with spatial weighting """
        super(Masked_LPIPS_Loss, self).__init__()
        self.lpips = lpips.LPIPS(net=net, spatial=True).eval()
        if use_gpu:
            self.lpips = nn.DataParallel(self.lpips).cuda()
        if precision == 'half':
            self.lpips.half()
        elif precision == 'float':
            self.lpips.float()
        elif precision == 'double':
            self.lpips.double()
        return

    def forward(self, im0, im1, w=None):
        """ ims have dimension BCHW while mask is B1HW """
        check_loss_input(im0, im1, w)
        # lpips takes the sum of each spatial map
        loss = self.lpips(im0, im1)
        if w is not None:
            n = torch.sum(loss * w, [1, 2, 3])
            d = torch.sum(w, [1, 2, 3])
            loss = n / d
        return loss

    def __call__(self, im0, im1, w=None):
        return self.forward(im0, im1, w)


class Masked_L1_Loss(nn.Module):
    def __init__(self):
        super(Masked_L1_Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, pred, ref, w=None):
        """ ims have dimension BCHW while mask is B1HW """
        check_loss_input(pred, ref, w)
        loss = self.loss(pred, ref)
        assert(pred.shape[1] == ref.shape[1])
        channels = pred.shape[1] 
        if w is not None:
            w = w.repeat(1, channels, 1, 1) # repeat on channel wise dim
            n = torch.sum(loss * w, [1, 2, 3])
            d = torch.sum(w, [1, 2, 3])
            loss = n / d
        return loss

class Masked_MSE_Loss(nn.Module):
    def __init__(self):
        super(Masked_MSE_Loss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, pred, ref, w=None):
        """ ims have dimension BCHW while mask is B1HW """
        check_loss_input(pred, ref, w)
        loss = self.loss(pred, ref)
        assert(pred.shape[1] == ref.shape[1])
        channels = pred.shape[1] 
        if w is not None:
            w = w.repeat(1, channels, 1, 1) # repeat on channel wise dim
            n = torch.sum(loss * w, [1, 2, 3])
            d = torch.sum(w, [1, 2, 3])
            loss = n / d
        return loss
