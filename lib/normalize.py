import torch
from torch.autograd import Variable
from torch import nn

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, imgs):
        norm = imgs.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = imgs.div(norm)
        return out
