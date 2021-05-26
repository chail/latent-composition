from abc import ABC, abstractmethod
from utils import zdataset
import numpy as np
import torch

def define_nets(nettype, domain, use_RGBM=True, use_VAE=False,
                ckpt_path='pretrained', load_encoder=True, device='cuda'):
    assert(not (use_RGBM and use_VAE)) # check that at most 1 is true
    if nettype == 'proggan':
        return ProgganNets(domain, use_RGBM, use_VAE, ckpt_path, load_encoder, device)
    elif nettype == 'stylegan':
        return StyleganNets(domain, use_RGBM, use_VAE, ckpt_path, load_encoder, device)
    else:
        raise NotImplementedError

class Nets(ABC):

    @abstractmethod
    def sample_zs(self, n, seed):
        pass

    @abstractmethod
    def zs2image(self, zs):
        pass

    @abstractmethod
    def seed2image(self, n, seed):
        pass

    @abstractmethod
    def encode(self, image, mask):
        pass

    @abstractmethod
    def decode(self, latent):
        pass

    @abstractmethod
    def invert(self, image, mask):
        pass

class ProgganNets(Nets):
    def __init__(self, domain, use_RGBM=True, use_VAE=False,
                 ckpt_path='pretrained', load_encoder=True, device='cuda'):
        from . import proggan_networks
        setting = proggan_networks.proggan_setting(domain)
        self.generator = proggan_networks.load_proggan(domain).to(device)
        if load_encoder:
            self.encoder = proggan_networks.load_proggan_encoder(
                domain, nz=setting['nz'], outdim=setting['outdim'],
                use_RGBM=use_RGBM, use_VAE=use_VAE,
                resnet_depth=setting['resnet_depth'],
                ckpt_path=ckpt_path).to(device)
        self.setting = setting
        self.device = device
        self.use_RGBM = use_RGBM
        self.use_VAE = use_VAE

    def sample_zs(self, n=100, seed=1, device=None):
        result = zdataset.z_sample_for_model(self.generator, n,
                                             seed).to(self.device)
        if device is None:
            result = result.to(self.device)
        else:
            result = result.to(device)
        return result

    def zs2image(self, zs):
        return self.generator(zs)

    def seed2image(self, n, seed):
        zs = self.sample_zs(n, seed)
        return self.zs2image(zs)

    def encode(self, image, mask=None):
        if mask is None:
            mask = torch.ones_like(image)[:, :1, :, :]
        if torch.max(mask) == 1:
            # pgan mask is [-0.5, 0.5]
            mask = mask - 0.5
        if self.use_RGBM or self.use_VAE:
            net_input = torch.cat([image, mask], dim=1)
        else:
            net_input = image

        encoded = self.encoder(net_input)

        if self.use_VAE:
            nz = encoded.shape[1] //2 # vae predicts mean and sigma
            sample = torch.randn_like(encoded[:, nz:, :, :])
            encoded_mean  = encoded[:, nz:, :, :]
            encoded_sigma = torch.exp(encoded[:, :nz, :, :])
            encoded = encoded_mean + encoded_sigma * sample
        return encoded

    def decode(self, latent):
        return self.zs2image(latent)

    def invert(self, image, mask=None):
        encoded = self.encode(image, mask)
        return self.decode(encoded)


class StyleganNets(Nets):
    def __init__(self, domain, use_RGBM=True, use_VAE=False, ckpt_path='pretrained', load_encoder=True, device='cuda'):
        from . import stylegan_networks
        setting = stylegan_networks.stylegan_setting(domain)
        self.generator = stylegan_networks.load_stylegan(
            domain, size=setting['outdim']).to(device)
        if load_encoder:
            self.encoder = stylegan_networks.load_stylegan_encoder(
                domain, nz=setting['nlatent'], outdim=setting['outdim'],
                use_RGBM=use_RGBM, use_VAE=use_VAE,
                resnet_depth=setting['resnet_depth'],
                ckpt_path=ckpt_path).to(device)
        self.setting = setting
        self.device = device
        self.use_RGBM = use_RGBM
        self.use_VAE = use_VAE

    def sample_zs(self, n=100, seed=1, device=None):
        depth = self.setting['nz']
        rng = np.random.RandomState(seed)
        result = torch.from_numpy(
                rng.standard_normal(n * depth)
                .reshape(n, depth)).float()
        if device is None:
            result = result.to(self.device)
        else:
            result = result.to(device)
        return result

    def zs2image(self, zs):
        ws = self.generator.gen.style(zs)
        return self.generator(ws)

    def seed2image(self, n, seed):
        zs = self.sample_zs(n, seed)
        return self.zs2image(zs)

    def encode(self, image, mask=None):
        if mask is None:
            mask = torch.ones_like(image)[:, :1, :, :]
        # stylegan mask is [0, 1]
        if torch.min(mask) == -0.5 :
            mask += 0.5
        assert(torch.min(mask >= 0))

        if self.use_RGBM or self.use_VAE:
            net_input = torch.cat([image, mask], dim=1)
        else:
            net_input = image

        encoded = self.encoder(net_input)
        # encoder shape = [batch_size, layers, nets.setting['nz']]

        if self.use_VAE:
            nlayers = self.setting['nlatent'] // self.setting['nz']
            assert(encoded.shape[1] == 2*nlayers)
            sample = torch.randn_like(encoded[:, nlayers:, :])
            encoded_mean  = encoded[:, nlayers:, :]
            encoded_sigma = torch.exp(encoded[:, :nlayers, :])
            encoded = encoded_mean + encoded_sigma * sample

        return encoded

    def decode(self, latent):
        return self.generator(latent)

    def invert(self, image, mask=None):
        encoded = self.encode(image, mask)
        return self.decode(encoded)
