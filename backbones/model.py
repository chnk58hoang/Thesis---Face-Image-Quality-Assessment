# from backbones.iresnet import iresnet50
from torchvision.models import resnet50
import torch.nn as nn
import torch
import pickle


def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            own_state[name].copy_(torch.from_numpy(param))


class Explainable_FIQA(nn.Module):
    def __init__(self, train_q_only=False, weight_path=None):
        super().__init__()
        self.backbone = nn.Sequential(*(list(resnet50().children())[:-1]))
        load_state_dict(self.backbone, fname=weight_path)
        self.medium = nn.Sequential(*[nn.LazyLinear(256), nn.LazyLinear(64)])
        self.pose = nn.Sequential(*[nn.LazyLinear(32), nn.LazyLinear(16), nn.LazyLinear(2)])

    def forward(self, x):
        feature = self.backbone(x)
        feature = feature.reshape(feature.size(0), -1)
        medium = self.medium(feature)
        qscore = self.pose(medium)
        return qscore
