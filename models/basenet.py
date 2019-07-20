import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .resnet import *
import numpy as np

class BaseNet(nn.Module):

    def __init__(self, num_classes=152):
        super(BaseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x,inplace=True)
        x = self.classifier(x)
        return x

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class MobileNet(nn.Module):

    def __init__(self,num_classes = 152):
        super(MobileNet,self).__init__()
        self.features = nn.Sequential(
            conv_bn(3,16,2),
            conv_dw(16,32,2),
            conv_dw(32,64,2),
            conv_dw(64,128,2),
        )
        self.classifier = nn.Linear(256*128,num_classes)

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x,inplace=False)
        x = self.classifier(x)
        return x


class SqueezeNet(nn.Module):
    def __init__(self,num_classes):
        super(SqueezeNet,self).__init__()
        self.pretrain_net = models.squeezenet1_1(pretrained=True)
        self.base_net = self.pretrain_net.features
        self.pooling  = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * 2, num_classes)
    def forward(self,x):
        x = self.base_net(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class resnet(nn.Module):
    def __init__(self, num_classes):
        super(resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        orig_resnet = resnet50(pretrained=True)
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        self.pooling = nn.AdaptiveMaxPool2d((1,1))
        self.conv4 = nn.Conv2d(2048,512,(8, 4), groups=16)
        self.relu4 = nn.ReLU()
        #self.conv5 = nn.Conv2d(1024,128,(16, 8), groups=16)
        #self.relu5 = nn.ReLU()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        p4 = self.relu4(self.conv4(c4))
        #p3 = self.relu5(self.conv5(c3))
        #p = torch.cat((p3,p4),dim=1)
        p = p4
        p = p.view(p.size(0), -1)
        out = self.fc(p)
        return out