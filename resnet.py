import torch, pdb
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5,
                               stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential( nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        self.activation1 = nn.GELU()

    def forward(self, x, lastlayer = 0):
        out = self.activation1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        return out


class ResNetTx(nn.Module):
    def __init__(self):
        super(ResNetTx, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = ResBlock(64, 128, stride=1)
        self.layer2 = ResBlock(128, 128, stride=2)
        self.layer3 = ResBlock(128, 32, stride=1)
        self.layer4 = ResBlock(32, 8, stride=2)
        self.activation1 = nn.GELU()

    def forward(self, x):
        # out = self.bn1(self.conv1(x))
        out = self.activation1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out, lastlayer = 1)
        return out

class ResNetRx(nn.Module):
    def __init__(self):
        super(ResNetRx, self).__init__()
        self.conv1 = nn.Conv2d(8, 64, kernel_size=9, stride=1, padding=4, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = ResBlock(64, 128, stride=1)
        self.layer1_ = nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2,padding=2,output_padding =1)
        self.layer2 = ResBlock(128, 128, stride=1)
        self.layer2_ = nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2,padding=2,output_padding =1)
        self.layer3 = ResBlock(128, 64, stride=1)
        self.layer4 = ResBlock(64, 3, stride=1)
        self.activation1 = nn.GELU()

    def forward(self, x):
        out = self.activation1(self.bn1(self.conv1(x)))
        if verbose == 1:
            print('encoder arch')
            print(x.shape)
            print(out.shape)
        out = self.layer1(out)
        out = self.layer1_(out)
        out = self.layer2(out)
        out = self.layer2_(out)
        out = self.layer3(out)
        out = self.layer4(out, lastlayer = 1)
        return out



