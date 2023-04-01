import torch
from backbones.model import ExplainableFIQA

model = ExplainableFIQA()
for m in model.modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.track_running_stats = False
        m.affine = True

