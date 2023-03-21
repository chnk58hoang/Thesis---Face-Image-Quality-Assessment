import torch.nn as nn
from backbones.iresnet import iresnet100
import torch

class ExplainFIQA(nn.Module):
    def __init__(self,weigth_path):
        super().__init__()
        self.backbone = iresnet100()
        self.backbone.load_state_dict(torch.load(weigth_path,map_location='cpu'),strict=False)
        self.pose_classifier = nn.Sequential(*[nn.LazyLinear(128),nn.LazyLinear(32),nn.LazyLinear(7)])

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self,x):
        fea,qs = self.backbone(x)
        pose = self.pose_classifier(fea)
        return fea,qs,pose