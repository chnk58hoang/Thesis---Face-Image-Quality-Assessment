from torch.nn import functional as F
import torch.nn as nn
import torch


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SharpFace(nn.CrossEntropyLoss):
    def __init__(self, in_feature, num_classes, s):
        super().__init__()
        self.in_feature = in_feature
        self.num_classes = num_classes
        self.s = s
        self.kernel = nn.Parameter(torch.FloatTensor(in_feature, num_classes))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embeddings, labels, sharpness):
        """
        :param embedding: (batch_size,512)
        :param label: (batch_size,num_classes)
        :return:
        """
        embeddings = l2_norm(embeddings, axis=1)
        kernel = l2_norm(self.kernel, axis=1)
        cos_theta = torch.mm(embeddings, kernel)
        labels = F.one_hot(labels, num_classes=self.num_classes)
        cos_theta = cos_theta.clamp(-1, 1)
        index = torch.where(labels == 1.0)[1].unsqueeze(1)
        m_hot = torch.zeros(index.size()[0], self.num_classes, device=cos_theta.device)
        sharpness = sharpness.unsqueeze(1)
        m_hot.scatter_(1, index, sharpness)
        cos_theta.acos_()
        cos_theta += m_hot
        cos_theta.cos_().mul_(self.s)
        float_label = torch.tensor(labels.clone().detach(), dtype=float)
        loss = super().forward(cos_theta, float_label)
        return loss


if __name__ == '__main__':
    sharpface = SharpFace(10, 10, s=0.5)
    embeddings = torch.rand(4, 10)
    labels = torch.tensor([1, 4, 3, 5])
    sharp = torch.FloatTensor([0.5, 0.6, 0.65, 0.5])
    print(sharpface(embeddings, labels, sharp))
