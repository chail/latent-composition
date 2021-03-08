import torch
from collections import OrderedDict
from . import util, pbar
from torch.nn.functional import l1_loss, mse_loss
from . import losses

masked_l1_loss = losses.Masked_L1_Loss()
masked_lpips_loss = losses.Masked_LPIPS_Loss()

def invert_lbfgs(nets, target, mask=None, lambda_f=0.25, lambda_l=0.5,
                 num_steps=3000, initial_latent=None):
    from . import LBFGS
    G = nets.generator
    E = nets.encoder
    G.eval()
    E.eval()
    G.cuda()
    E.cuda()

    if mask is None:
        # mask is just all ones (no mask)
        mask = torch.ones_like(target)[:, :1, : :]

    torch.set_grad_enabled(False)

    true_x = target.cuda()
    mask = mask.cuda()
    init_z = nets.encode(true_x, mask)
    if initial_latent is not None:
        assert lambda_f == 0.0
        init_z = initial_latent
    current_z = init_z.clone()
    target_x = target.clone().cuda()
    target_f = nets.encode(target.cuda(), mask)
    parameters = [current_z]

    util.set_requires_grad(False, G, E)
    util.set_requires_grad(True, *parameters)
    optimizer = LBFGS.FullBatchLBFGS(parameters)

    def compute_all_loss():
        current_x = nets.decode(current_z)
        all_loss = {}
        all_loss['x'] = ((1-lambda_l) * masked_l1_loss(current_x, target_x, mask) + lambda_l * masked_lpips_loss(current_x, target_x, mask))
        all_loss['z'] = 0.0 if not lambda_f else (
            mse_loss(target_f, nets.encode(current_x, mask)) * lambda_f)
        return current_x, all_loss

    def closure():
        optimizer.zero_grad()
        _, all_loss = compute_all_loss()
        return sum(all_loss.values())

    losses = []
    iterator = pbar(range(num_steps + 1))
    with torch.enable_grad():
        for step_num in iterator:
            if step_num == 0:
                loss = closure()
                loss.backward()
            else:
                options = {'closure': closure, 'current_loss': loss,
                        'max_ls': 10}
                loss, _, lr, _, _, _, _, _ = optimizer.step(options)
            losses.append(loss.detach().cpu().numpy())
    # get final results
    with torch.no_grad():
        current_x, all_loss = compute_all_loss()
    checkpoint_dict = OrderedDict(all_loss)
    checkpoint_dict['loss'] = sum(all_loss.values()).item()
    checkpoint_dict['init_z'] = init_z
    checkpoint_dict['target_x'] = target_x
    checkpoint_dict['current_z'] = current_z
    checkpoint_dict['current_x'] = current_x
    return checkpoint_dict, losses
