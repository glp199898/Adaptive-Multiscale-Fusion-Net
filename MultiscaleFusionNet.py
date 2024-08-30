import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torchsummary import summary
from SCConv import SCConv
import torch.nn.functional as f


__all__ = ['MultiscaleFusionNet']


class MFReLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):

        y = torch.where(torch.where(x + 1 < 0, 0, x - 0.5) > 0, torch.where(x - 0.5 < 0, 1/2*torch.sin(math.pi*x), x), torch.where(x + 1 < 0, 0, 1/2*torch.sin(math.pi*x)))

        return y

# ScconvSplitBlock

class ScconvSplitBlock(nn.Module):
    def __init__(self, in_channels, channels, padding=1, stride=1):
        super(ScconvSplitBlock, self).__init__()

        self.scconv = nn.Sequential(SCConv(in_channels, channels * 2, stride=stride, padding=padding, dilation=1, groups=2, pooling_r=4),
                                        BatchNorm2d(channels * 2),
                                        ReLU(inplace=True))

        self.channel_att = nn.Sequential(
            nn.Conv2d(channels, int(channels // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(channels // 4), channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.global_att = eca_block(channel=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x = self.scconv(x)
        batch_size, r_channels = x.shape[:2]
        splits = torch.split(x, int(r_channels / 2), dim=1)
        x = sum(splits)
        out = x + residual
        xc = self.channel_att(out)
        xg = self.global_att(out)
        x = xc + xg
        weight = self.sigmoid(x)
        output = 2 * x * weight + 2 * residual * (1 - weight)

        return output.contiguous()


# ScconvSplitBlockMFReLU

class ScconvSplitBlockMFReLU(nn.Module):
    def __init__(self, in_channels, channels, padding=1, stride=1):
        super(ScconvSplitBlockMFReLU, self).__init__()

        self.scconv = nn.Sequential(SCConv(in_channels, channels * 2, stride=stride, padding=padding, dilation=1, groups=2, pooling_r=4),
                                    BatchNorm2d(channels * 2), MFReLU())

        self.channel_att = nn.Sequential(
                         nn.Conv2d(channels, int(channels // 4), kernel_size=1, stride=1, padding=0),
                         nn.BatchNorm2d(int(channels // 4)),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(int(channels // 4), channels, kernel_size=1, stride=1, padding=0),
                         nn.BatchNorm2d(channels)
                         )

        self.global_att = eca_block(channel=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x = self.scconv(x)
        batch_size, r_channels = x.shape[:2]
        splits = torch.split(x, int(r_channels / 2), dim=1)
        x = sum(splits)
        out = x + residual
        xc = self.channel_att(out)
        xg = self.global_att(out)
        x = xc + xg
        weight = self.sigmoid(x)
        output = 2 * x * weight + 2 * residual * (1 - weight)

        return output.contiguous()


# FusionBasicBlock

class FusionBasicBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, MFReLU=True):
        super(FusionBasicBlock, self).__init__()

        if not MFReLU:
            self.conv1 = nn.Sequential(Conv2d(in_planes, planes * 4, kernel_size=(1, 1), stride=(1, 1)),
                                       BatchNorm2d(planes * 4), ReLU(inplace=True))
            self.conv2 = ScconvSplitBlock(planes * 4, planes * 4, stride=stride, padding=1)
            self.conv3 = nn.Sequential(Conv2d(planes * 4, planes, kernel_size=(1, 1), stride=(1, 1)),
                                       ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(Conv2d(in_planes, planes * 4, kernel_size=(1, 1), stride=(1, 1)),
                                       BatchNorm2d(planes * 4), ReLU(inplace=True))
            self.conv2 = ScconvSplitBlockMFReLU(planes * 4, planes * 4, stride=stride, padding=1)
            self.conv3 = nn.Sequential(Conv2d(planes * 4, planes, kernel_size=(1, 1), stride=(1, 1)),
                                       ReLU(inplace=True))

    def forward(self, x):

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        out = torch.cat([x, out], 1)

        return out

# MultiscaleLayer

class MultiscaleLayer(nn.Module):
    def __init__(self, inplace, plance):
        super(MultiscaleLayer, self).__init__()

        self.stem1 = nn.Sequential(Conv2d(inplace, inplace, kernel_size=1, stride=2, padding=0), nn.BatchNorm2d(inplace), nn.ReLU(inplace=True))
        self.stem2 = nn.Sequential(Conv2d(2 * inplace, plance, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(plance), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        out = self.stem1(x)
        branch2 = self.pool(x)
        out = torch.cat([branch2, out], 1)
        out = self.stem2(out)

        return out

# Transition

class Transition(nn.Module):
    def __init__(self, inplace, plance):
        super(Transition, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.Conv2d(in_channels=inplace, out_channels=plance, kernel_size=1, stride=1, padding=0, bias=False, dilation=2),
            nn.BatchNorm2d(plance),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):

        x = self.transition_layer(x)

        return x


# ECA block

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv_mask = nn.Conv2d(channel, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        input_x = input_x.view(batch, channel, height * width)
        input_x = input_x.unsqueeze(1)
        context_mask = self.conv_mask(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.unsqueeze(-1)
        context = torch.matmul(input_x, context_mask)
        x = context.view(batch, channel, 1, 1)

        return x


    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# SCConv

class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r):
        super(SCConv, self).__init__()

        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(inplanes, int(planes/2), kernel_size=3, stride=1, padding=padding, dilation=dilation, groups=groups, bias=False),
                    BatchNorm2d(int(planes/2)))

        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, int(planes/2), kernel_size=3, stride=1, padding=padding, dilation=dilation, groups=groups, bias=False),
                    BatchNorm2d(int(planes/2)))

        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False),
                    BatchNorm2d(planes))

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, f.interpolate(self.k2(x), identity.size()[2:])))

        out = torch.mul(self.k3(x), out)

        out = self.k4(out)

        return out


# mfnet
class mfnet(nn.Module):
    def __init__(self, block, FusionBlocks, bottleneck_width=64, num_classes=1000):
        super(mfnet, self).__init__()

        self.bottleneck_width = bottleneck_width

        self.deep_stem = nn.Sequential(Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                       BatchNorm2d(64), ReLU(inplace=True),
                                       Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                       BatchNorm2d(64), ReLU(inplace=True),
                                       Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                       BatchNorm2d(64), ReLU(inplace=True),
                                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.in_channels = 64
        self.layer1 = self._make_stage(block, FusionBlocks[0], 1, channels=32)
        self.layer2 = Transition(128, 64)
        self.layer3 = self._make_stage(block, FusionBlocks[1], 1, channels=32, MFReLU=True)
        self.layer4 = MultiscaleLayer(256, 128)
        self.layer5 = self._make_stage(block, FusionBlocks[2], 1, channels=32, MFReLU=True)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128*block.expansion, num_classes)


    def forward(self, x):
        x = self.deep_stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _make_stage(self, block, FusionBlocks, stride, channels, MFReLU=False):
        strides = [stride]+[1]*(FusionBlocks-1)
        layers = []

        i = 0

        for stride in strides:

            i += 1

            if (len(strides) == 6 or len(strides) == 12) and i == 1:

                self.in_channels = self.in_channels // 2
                layers.append(block(self.in_channels, channels, stride, MFReLU))
                self.in_channels = self.in_channels + channels

            else:
                layers.append(block(self.in_channels, channels, stride, MFReLU))
                self.in_channels = self.in_channels + channels

        return nn.Sequential(*layers)

def MultiscaleFusionNet(num_classes: int = 1000):
    model = mfnet(FusionBasicBlock, [2, 6, 12], num_classes=num_classes)

    return model


