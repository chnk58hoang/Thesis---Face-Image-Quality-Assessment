from backbones.iresnet import iresnet50
import torch.nn as nn
import torch


class XFIQA(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        self.backbone = nn.Sequential(*list(iresnet50().children())[:-2])
        self.backbone.load_state_dict(torch.load(weight_path), strict=False)

        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.LazyLinear(128)
        self.fc3 = nn.LazyLinear(32)
        self.pose_classifier = nn.LazyLinear(7)

    def forward(self, image):
        image = self.backbone(image)
        image = image.reshape(image.size(0), -1)
        image = self.fc1(image)
        image = self.fc2(image)
        pose = self.pose_classifier(image)
        return pose


