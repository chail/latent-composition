"""
Generator based on the unofficial implementation of StyleGAN2
"""

import json
import sys
import os
import torch
from torch import nn
from .PT_STYLEGAN2.model import Generator
from . import util

# We need this little hack to allow calling it as Generator(x)
class encapsulated_generator(nn.Module):
    def __init__(self, model, input_is_w=False):
        super().__init__()

        self.gen = model
        self.input_is_w = input_is_w

    def forward(self, x):
        #TODO: Check if x is array or not.
        return self.gen([x,], input_is_latent=self.input_is_w)[0]

def get_generator(filename, cfg=None, size=128, channel_multiplier=2):
    """
    Returns generator trained using the StyleGANv2 code.
    """

    latent = 512
    n_mlp = 8

    input_is_w = False
    if cfg is not None and cfg.optimize_to_w:
        input_is_w = True

    model = Generator(size, latent, n_mlp, channel_multiplier=channel_multiplier)
    if util.is_url(filename):
        model.load_state_dict(torch.hub.load_state_dict_from_url(filename)['g_ema'])
    else:
        model.load_state_dict(torch.load(filename)['g_ema'])

    G = encapsulated_generator(model, input_is_w=input_is_w)

    return G

def freeze_layers(generator, n_layers, freeze_front_layers=False):
    """
    Freeze layers (either first layers or last layers, depending on
    freeze_front_layers)
    """
    if freeze_front_layers:
        # Disable first n conv layers
        for param in generator.gen.conv1.parameters():
            param.requires_grad = False
        for l in generator.gen.convs[:n_layers]:
            print('Disabling layer')
            print(l)
            for param in l.parameters():
                param.requires_grad = False
    else:
        # Disable to_rgb layers, and last n convs
        for l in generator.gen.to_rgbs:
            print('Disabling layer')
            print(l)
            for param in l.parameters():
                param.requires_grad = False

        for l in generator.gen.convs[-n_layers:]:
            print('Disabling layer')
            print(l)
            for param in l.parameters():
                param.requires_grad = False



def train_layers(generator, train_layer_list):
    """
    Train each layer given in layer list.
    """
    for param in generator.parameters():
        param.requires_grad = False

    for layer in train_layer_list:
        for param in generator.gen.convs[layer].parameters():
            param.requires_grad = True

