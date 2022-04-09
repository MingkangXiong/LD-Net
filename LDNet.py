# https://github.com/ericsun99/Shufflenet-v2-Pytorch/blob/master/ShuffleNetV2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_, kaiming_normal_

from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class ASPP(nn.Module):
    
    def __init__(self, inplanes, planes, rate, dw=False):
        super(ASPP, self).__init__()
        self.rate=rate
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
            if dw:
                self.conv1 =SeparableConv2d(planes,planes,3,1,1)
            else:
                self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, bias=False,padding=1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu1 = nn.ReLU()
   
            #self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
            #                         stride=1, padding=padding, dilation=rate, bias=False)
        self.atrous_convolution = SeparableConv2d(inplanes,planes,kernel_size,1,padding,rate)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        
        
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        #x = self.relu(x)
        if self.rate!=1:
            x=self.conv1(x)
            x=self.bn1(x)
            x=self.relu1(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
    
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        
        if self.benchmodel == 1:
            #assert inp == oup_inc
        	self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )                
        else:                  
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )        
    
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
          
    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)        

    def forward(self, x):
        if 1==self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2==self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)

def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )

def downsample_spconv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        SeparableConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        SeparableConv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )

def downsample_sf(in_planes, out_planes):
    # return nn.Sequential(
    #     InvertedResidual(in_planes, out_planes, 2, 2),
    #     InvertedResidual(in_planes, out_planes, 1, 1)
    # )
    return InvertedResidual(in_planes, out_planes, 2, 2)

def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


def sep_conv(in_planes, out_planes):
    return nn.Sequential(
        SeparableConv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )
    
class UpSampleLayer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(UpSampleLayer, self).__init__()
        # self.conv = nn.Sequential(SeparableConv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        #             nn.ReLU(inplace=True))
        # self.conv = SeparableConv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.conv = InvertedResidual(in_planes, out_planes, 1, 2)

    def forward(self,x):
        x = self.conv(x)
        x_temp1 = torch.cat((x, x), 1)
        x_temp2 = torch.cat((x_temp1, x_temp1), 1)
        x = F.pixel_shuffle(x_temp2, 2)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        return x

def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]

class LDNet(nn.Module):

    def __init__(self, alpha=10, beta=0.01):
        super(LDNet, self).__init__()

        self.alpha = alpha
        self.beta = beta

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_sf(3,              conv_planes[0])
        self.conv2 = downsample_sf(conv_planes[0], conv_planes[1])
        self.conv3 = downsample_sf(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_sf(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_sf(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_sf(conv_planes[4], conv_planes[5])
        # self.conv7 = downsample_sf(conv_planes[5], conv_planes[6])

        rates = [1, 3, 6, 9]
        # rates = [1, 5, 9]

        x_c = conv_planes[5]
        out_c = 48

        self.aspp1 = ASPP(x_c, out_c, rate=rates[0], dw=True)
        self.aspp2 = ASPP(x_c, out_c, rate=rates[1], dw=True)
        self.aspp3 = ASPP(x_c, out_c, rate=rates[2], dw=True)
        self.aspp4 = ASPP(x_c, out_c, rate=rates[3], dw=True)

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        
        # self.upconv7 = UpSampleLayer(conv_planes[6],   upconv_planes[0])
        self.upconv6 = UpSampleLayer(x_c + 4 * out_c,                   upconv_planes[1])
        self.upconv5 = UpSampleLayer(upconv_planes[1] + conv_planes[4], upconv_planes[2])
        self.upconv4 = UpSampleLayer(upconv_planes[2] + conv_planes[3], upconv_planes[3])
        self.upconv3 = UpSampleLayer(upconv_planes[3] + conv_planes[2], upconv_planes[4])
        self.upconv2 = UpSampleLayer(upconv_planes[4] + conv_planes[1], upconv_planes[5])
        self.upconv1 = UpSampleLayer(upconv_planes[5] + conv_planes[0], upconv_planes[6])

        self.predict_disp1 = predict_disp(upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)

        x1 = self.aspp1(out_conv6)
        x2 = self.aspp2(out_conv6)
        x3 = self.aspp3(out_conv6)
        x4 = self.aspp4(out_conv6)

        out_conv6 = torch.cat((out_conv6, x1, x2, x3, x4), dim=1)

        out_upconv6 = self.upconv6(out_conv6)

        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        
        out_upconv5 = self.upconv5(concat6)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)

        out_upconv4 = self.upconv4(concat5)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)

        out_upconv3 = self.upconv3(concat4)
        concat3 = torch.cat((out_upconv3, out_conv2), 1)

        out_upconv2 = self.upconv2(concat3)
        concat2 = torch.cat((out_upconv2, out_conv1), 1)

        out_upconv1 = self.upconv1(concat2)
        concat1 = out_upconv1

        disp1 = self.alpha * self.predict_disp1(concat1) + self.beta

        if self.training:
            return disp1, disp1, disp1, disp1
        else:
            return disp1