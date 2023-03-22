from backbones.iresnet import iresnet100
import torch.nn as nn



model = iresnet100(pose=True)
print(model)