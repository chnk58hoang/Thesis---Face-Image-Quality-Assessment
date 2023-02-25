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


class PoseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.LazyLinear(32)
        self.fc2 = nn.LazyLinear(16)
        self.classifier = nn.LazyLinear(2)

    def forward(self, x):
        x = self.fc1(x)
        pose_represent = self.fc2(x)
        pose = self.classifier(pose_represent)
        return pose_represent, pose


class BrightClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.LazyLinear(32)
        self.fc2 = nn.LazyLinear(16)
        self.classifier = nn.LazyLinear(2)

    def forward(self, x, pose_represent):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.cat([x, pose_represent], dim=1)
        br = self.classifier(x)
        return br


class Explainable_FIQA(nn.Module):
    def __init__(self, train_q_only=False, weight_path=None):
        super().__init__()
        self.backbone = nn.Sequential(*(list(resnet50().children())[:-1]))
        load_state_dict(self.backbone, fname=weight_path)
        self.medium = nn.Sequential(*[nn.LazyLinear(256), nn.LazyLinear(64)])
        self.pose_classifier = PoseClassifier()
        self.br_classifier = BrightClassifier()

    def forward(self, x):
        feature = self.backbone(x)
        feature = feature.reshape(feature.size(0), -1)
        root = self.medium(feature)
        pose_represent, pose = self.pose_classifier(root)
        br = self.br_classifier(root, pose_represent)
        return pose, br


if __name__ == '__main__':
    x = torch.load('best_model.pth')
    print(x)