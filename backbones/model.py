from backbones.iresnet import IBasicBlock, conv1x1
import torch.nn as nn
import torch


class PoseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.dilation = 1
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(num_parameters=64)

        self.layer1 = self._make_layer(IBasicBlock, 64, 3, stride=2, use_se=True)
        self.layer2 = self._make_layer(IBasicBlock, 128, 5, stride=2, use_se=True)
        self.layer3 = self._make_layer(IBasicBlock, 256, 3, stride=2, use_se=True)
        self.layer4 = self._make_layer(IBasicBlock, 512, 3, stride=2, use_se=True)

        self.classifier = nn.Sequential(*[nn.LazyLinear(64,bias=False),nn.LazyLinear(32,bias=False), nn.LazyLinear(7,bias=False)])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_se=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, use_se=use_se))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation, use_se=use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)
        pose = self.classifier(x)
        return pose
