from . import customnet
import torch
import torch.nn as nn
from torchvision import models as tv

def alexnet():
    def change_out(layers):
        ind, layer = [(i, l) for i, (n, l) in enumerate(layers) if n == 'fc7'][0]
        layers[ind+1:] = []
        return layers
    model = customnet.CustomAlexNet(modify_sequence=change_out)
    from torchvision.models.alexnet import model_urls
    import torch.utils.model_zoo as model_zoo
    model.load_state_dict(model_zoo.load_url(model_urls['alexnet']),
                          strict=False)
    return model

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        self.orig_model = tv.vgg16(pretrained=pretrained)
        self.new_classifier = nn.Sequential(
            *list(self.orig_model.classifier.children())[:4]
        )

    def forward(self, x):
        x = self.orig_model.features(x)
        x = self.orig_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.new_classifier(x)
        return x

