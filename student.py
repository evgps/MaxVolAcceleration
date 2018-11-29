import torch
import torch.nn as nn
from copy import deepcopy

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x
    
class Tensorise(nn.Module):
    def __init__(self, shape):
        super(Tensorise, self).__init__()
        self.shape = shape
    def forward(self, x):
        x = x.reshape(x.shape[0], self.shape[0], self.shape[1], self.shape[2])
        return x
        
class MaxVolMaxPool(nn.Module):
    def forward(self, x):
        x = x.view(x.shape[0], -1, 2).squeeze(-1)
        return torch.nn.functional.max_pool2d(x, 2).view(x.shape[0], -1)

class Bias(nn.Module):
    def __init__(self, bias):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(bias)
    def forward(self, x):
        return x + self.bias

class Student(nn.Module):
    def __init__(self, compressed_model, device='cuda'):
        super(Student, self).__init__()
        self.device = device
        self.features = []
        module_list = []
        for idx, (W, bias, has_maxpool, V, block, mode, shape) in enumerate(
                                        zip(compressed_model.W,
                                        compressed_model.biases,
                                        compressed_model.has_maxpool,
                                        compressed_model.V,
                                        compressed_model.blocks,
                                        compressed_model.mode,
                                        compressed_model.shapes)):
            if mode is None:
                self.features += [block]
            else:
                if mode == 'first':
                    self.features += [Flatten()]
                self.features += [torch.nn.Linear(*W.shape[::-1], bias=False)]
                self.features[-1].weight.data = W.to(self.device)
                if has_maxpool:
                    self.features += [MaxVolMaxPool()]
                if bias is not None and torch.norm(bias) > 1e-5:
                    self.features += [Bias(bias.view(-1).to(self.device))]
                self.features += [torch.nn.ReLU()]
                if mode == 'last':
                    self.features += [torch.nn.Linear(*V.shape[::-1], bias=False)]
                    self.features[-1].weight.data = nn.Parameter(V.to(self.device))
                    self.features += [Tensorise(shape)]
        self.features = nn.Sequential(*self.features)
        self.classifier = deepcopy(compressed_model.classifier)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x