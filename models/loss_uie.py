import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from models.ssim import *

ssim_loss = SSIMLoss(11)

class loss_uie(nn.Module):
    def __init__(self):
        super(loss_uie, self).__init__()
        self.gradient = kornia.filters.SpatialGradient()
    def forward(self, E, A):
        loss_grad=F.l1_loss(self.gradient(E),self.gradient(A))
        total_loss = 40*F.l1_loss(E,A)+40*loss_grad+100*ssim_loss(E,A)
        
        return total_loss
