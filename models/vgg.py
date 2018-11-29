'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import os.path
import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, pretrained=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        if pretrained:
            print('Load pretrained weights')
            assert os.path.exists(pretrained)
            tmp = torch.load(pretrained)
            self.load_state_dict(tmp)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False, pooling = 'MAX', slope = 0):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            if pooling == 'AVG':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(slope, inplace=True)]
            else:
                layers += [conv2d, nn.LeakyReLU(slope, inplace=True)]
       
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), pretrained=pretrained)


def vgg11_bn(pretrained=False):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), pretrained=pretrained)


def vgg13(pretrained=False):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), pretrained=pretrained)


def vgg13_bn(pretrained=False):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True), pretrained=pretrained)


def vgg16(pretrained=False):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), pretrained=pretrained)


def vgg16_bn(pretrained=False):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), pretrained=pretrained)

def vgg16_avg(pretrained=False):
    """VGG 16-layer model (configuration "D") with average pooling"""
    return VGG(make_layers(cfg['D']), pooling = 'AVG', slope = 0.01)


def vgg19(pretrained=False):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), pretrained=pretrained)


def vgg19_bn(pretrained=False):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True), pretrained=pretrained)


def vgg19_avg(pretrained=False):
    """VGG 19-layer model (configuration "E") with average pooling"""
    return VGG(make_layers(cfg['E']), pooling = 'AVG', slope = 0.01, pretrained=pretrained)
