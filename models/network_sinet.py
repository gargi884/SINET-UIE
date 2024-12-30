import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_theta(x,theta):
    return torch.mul(torch.sign(x), F.relu(torch.abs(x) - theta))

class BasicBlock(nn.Module):
    def __init__(self,num_channel=1,num_filter=64,kernel_size=9):
        super(BasicBlock, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=num_channel, out_channels=num_filter, kernel_size=kernel_size,
                                stride=1, padding='same', bias=False)
        nn.init.xavier_uniform_(self.conv_in.weight.data)
        self.ch_down = nn.Conv2d(in_channels=num_filter, out_channels=num_channel, kernel_size=kernel_size,
                                stride=1, padding='same', bias=False)
        nn.init.xavier_uniform_(self.ch_down.weight.data)
        self.ch_up = nn.Conv2d(in_channels=num_channel, out_channels=num_filter, kernel_size=kernel_size,
                                stride=1, padding='same', bias=False)
        nn.init.xavier_uniform_(self.ch_up.weight.data)
    def forward(self, I, x,theta):
        x1 = self.ch_down(x)
        x2 = self.ch_up(x1)
        x3 = x-x2+self.conv_in(I)
        return soft_theta(x3,theta)
        
class Prediction(nn.Module):
    def __init__(self, num_channel=1,num_filter=64,kernel_size=9, n_layer=4):
        super(Prediction, self).__init__()
        self.w_theta = nn.Parameter(torch.Tensor([-0.5]))
        self.b_theta = nn.Parameter(torch.Tensor([-2]))
        self.n_layer = n_layer
        self.conv_0 = nn.Conv2d(in_channels=num_channel, out_channels=num_filter, kernel_size=kernel_size,
                                stride=1, padding='same', bias=False)
        nn.init.xavier_uniform_(self.conv_0.weight.data)
        self.basicLayer = []
        count = 0
        for i in range(self.n_layer):
            count += 1
            self.basicLayer.append(BasicBlock(num_channel,num_filter, kernel_size))
        self.basicLayers = nn.ModuleList(self.basicLayer)
        self.Sp = nn.Softplus()

    def forward(self, Isi):
        theta_0 = self.Sp(self.b_theta)
        zi_0 = soft_theta(self.conv_0(Isi),theta_0)
        zi_k = zi_0
        for k in range(self.n_layer):
            theta_k = self.Sp(self.w_theta*k+self.b_theta)
            zi_k = self.basicLayers[k](Isi, zi_k, theta_k)
        return zi_k

class SINET(nn.Module):
    def __init__(self,num_channel=1,num_filter=16,kernel_size=11,n_layer=4):
        super(SINET, self).__init__()
        self.prediction1 = Prediction(num_channel=num_channel,num_filter=num_filter,kernel_size=kernel_size,n_layer=n_layer)
        self.conv_out1 = nn.Conv2d(in_channels=num_filter, out_channels=num_channel, kernel_size=kernel_size,
                                stride=1, padding='same', bias=False)
        nn.init.xavier_uniform_(self.conv_out1.weight.data)
        self.prediction2 = Prediction(num_channel=num_channel,num_filter=num_filter,kernel_size=kernel_size,n_layer=n_layer)
        self.conv_out2 = nn.Conv2d(in_channels=num_filter, out_channels=num_channel, kernel_size=kernel_size,
                                stride=1, padding='same', bias=False)
        nn.init.xavier_uniform_(self.conv_out2.weight.data)
        self.prediction3 = Prediction(num_channel=num_channel,num_filter=num_filter,kernel_size=kernel_size,n_layer=n_layer)
        self.conv_out3 = nn.Conv2d(in_channels=num_filter, out_channels=num_channel, kernel_size=kernel_size,
                                stride=1, padding='same', bias=False)
        nn.init.xavier_uniform_(self.conv_out3.weight.data)
    def forward(self, x):
        x1, x2, x3 = x.chunk(3, dim=1)
        x1 = self.conv_out1(self.prediction1(x1))
        x2 = self.conv_out2(self.prediction2(x2))
        x3 = self.conv_out3(self.prediction3(x3))
        x=torch.cat([x1, x2,x3], dim=1)
        return x    
