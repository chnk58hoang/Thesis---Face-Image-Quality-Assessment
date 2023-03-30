from backbones.iresnet import iresnet100
import torch.nn as nn
import torch


class ExplainableFIQA(nn.Module):
    def __init__(self, backbone_weight):
        super().__init__()
        self.backbone = iresnet100()
        self.pose_classifier = nn.Sequential(
            *[nn.LazyLinear(256), nn.LazyLinear(128), nn.LazyLinear(64), nn.LazyLinear(32), nn.LazyLinear(7)])

        self.backbone.load_state_dict(torch.load(backbone_weight, map_location='cpu'))

    def forward(self, x):
        emb, qs = self.backbone(x)
        pose = self.pose_classifier(emb)
        return emb, qs, pose
