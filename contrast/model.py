# -*- codding: utf-8 -*-
'''
@Author : Yuren
@Dare   : 2021/12/20-9:50 下午
'''
import torch
from torchsummary import summary
import torch.nn as nn
class DenseLayer(nn.Module):
    def __init__(self,in_channel,bn_size,growthrate):
        super(DenseLayer, self).__init__()
        self.den = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channel,out_channels = bn_size*growthrate,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(bn_size*growthrate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size*growthrate,out_channels=growthrate,kernel_size=3,stride=1,padding=1)
        )
    def forward(self,x):
        out = self.den(x)
        out = torch.cat([x,out],dim=1)
        return out

class DenseBlock(nn.Module):
    def __init__(self,in_channels,nums,bn_size,growthrate):
        super(DenseBlock, self).__init__()
        self.db = self._makedense(DenseLayer,in_channels,nums,bn_size,growthrate)
    def _makedense(self,block,in_channels,nums,bn_size,growthrate):
        layers = []
        for i in range(nums):
            layer = block(in_channels+i*growthrate, bn_size, growthrate)
            layers.append(layer)
        return nn.Sequential(*layers)
    def forward(self,x):
        out = self.db(x)
        return out


class TransitionLayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(TransitionLayer, self).__init__()
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
    def forward(self,x):
        out = self.trans(x)
        return out


class Prl(nn.Module):
    def __init__(self):
        super(Prl, self).__init__()
        self.conv = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.DenseBlock1 = DenseBlock(64,6,1,128)
        self.Trans1 = TransitionLayer(832,128)
        self.DenseBlock2 = DenseBlock(128,12,1,128)
        self.Trans2 = TransitionLayer(1664,256)
        self.DenseBlock3 = DenseBlock(256,24,1,128)
        self.Trans3 = TransitionLayer(3328,512)
        self.DenseBlock4 = DenseBlock(512,16,1,128)
        self.Trans4 = TransitionLayer(2560,1024)
        self.GMP=nn.AdaptiveMaxPool2d(1)
        self.linear1 = nn.Linear(1024,202)
        self.linear2 = nn.Linear(1024,202)
        self.linear3 = nn.Linear(1024,202)
        self.linear4 = nn.Linear(1024,202)

    def forward(self,x):
        out = self.conv(x)  # 1 * 64 * 256 * 256
        out = self.pool(out) # 1 * 64 * 128 * 128

        out = self.DenseBlock1(out)
        out = self.Trans1(out)# 1 * 128 * 64 * 64

        out = self.DenseBlock2(out)
        out = self.Trans2(out)# 1 * 256 * 32 * 32
        out = self.DenseBlock3(out)
        out = self.Trans3(out)# 1 * 512 * 16 * 16
        out = self.DenseBlock4(out)
        out = self.Trans4(out)# 1 * 1024 * 8 * 8
        #print(out.shape)
        out = self.GMP(out)   # 1 * 1024
        out = out.reshape(x.size(0),1,1024)
        #print(out.shape)
        out1 = self.linear1(out) # 1 * 202
        out2 = self.linear2(out)# 1 * 202
        out3 = self.linear3(out)# 1 * 202
        out4 = self.linear4(out)# 1 * 202
        out1 = out1.squeeze()
        out2 = out2.squeeze()
        out3 = out3.squeeze()
        out4 = out4.squeeze()
        return [out1,out2,out3,out4]

if __name__ == '__main__':
    x = torch.randn([2,1,512,512])
    net = Prl()
    print(net(x)[0].shape)
    #summary(net,(1,512,512),batch_size=1,device='cpu')




