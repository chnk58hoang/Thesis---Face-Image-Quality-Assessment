from backbones.iresnet import iresnet100, conv1x1, IBasicBlock
import torch.nn as nn
import torch


class ExplainableFIQA(nn.Module):
    def __init__(self, backbone_weight=None, num_classes=7):
        super().__init__()
        self.fp16 = False
        self.dilation = 1
        self.inplanes = 512
        self.groups = 1
        self.base_width = 64
        self.backbone = iresnet100()
        self.class_branch1 = self._make_layer(IBasicBlock, 512, 1, stride=2, use_se=True)
        self.class_branch2 = self._make_layer(IBasicBlock, 128, 1, stride=2, use_se=True)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pose_classifier = nn.Sequential(
            *[self.class_branch1, self.class_branch2, nn.Flatten(start_dim=1), self.fc1, self.fc2])
        self.backbone.load_state_dict(torch.load(backbone_weight, map_location='cpu'))

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
        with torch.cuda.amp.autocast(self.fp16):
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.prelu(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            pose = self.pose_classifier(x)
            x = self.backbone.bn2(x)
            x = torch.flatten(x, 1)
            x = self.backbone.dropout(x)
        x = self.backbone.fc(x.float() if self.fp16 else x)
        x = self.backbone.features(x)
        qs = self.backbone.qs(x)
        return x, qs, pose
