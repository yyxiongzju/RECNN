'''RECNN in PyTorch.

See the paper "On Training Deep 3D CNN Models with Dependent Samples in Neuroimaging" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block3D(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block3D, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu1 = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu2 = nn.ReLU6(inplace=True)
        self.conv3 = nn.Conv3d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(out_planes),
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class RECNN(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  16, 2, 2),
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6, 128, 3, 2)]

    def __init__(self, scale=1.0, input_size=224, num_classes=2):
        super(RECNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(8)
        self.layers = self._make_layers(in_planes=8)
        self.conv2 = nn.Conv3d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(512)
        self.linear = nn.Linear(512, num_classes)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block3D(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class RECNN_Extractor(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  16, 2, 2),
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6, 128, 3, 2)]

    def __init__(self, scale=1.0, input_size=224, num_classes=2):
        super(RECNN_Extractor, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(8)
        self.layers = self._make_layers(in_planes=8)
        self.conv2 = nn.Conv3d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(512)
        self.linear = nn.Linear(512, num_classes)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block3D(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out

class RECNN_Mask(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  16, 2, 2),
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6, 128, 3, 2)]

    def __init__(self, scale=1.0, input_size=224, num_classes=2):
        super(RECNN_Mask, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(8)
        self.layers = self._make_layers(in_planes=8)
        self.conv2 = nn.Conv3d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(512)
        self.linear = nn.Linear(512, num_classes)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block3D(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        return out