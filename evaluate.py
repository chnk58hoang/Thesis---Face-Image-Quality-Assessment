import numpy as np
import torch
import argparse
from backbones.iresnet import iresnet100
import glob
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

t = transforms.Compose([transforms.ToTensor(), transforms.Resize(112)])


def process_grad(gradient, a=10 ** 7.5, b=2):
    grad = abs(gradient)
    x = np.mean(grad, axis=0)
    x = 1 - (1 / (1 + (a * x ** b)))
    return x


def get_fnmr(model1, pivot, q_threshold, s_threshold=0.3):
    piv_img = Image.open(pivot)
    piv_img = t(piv_img)
    piv_img.requires_grad = True
    piv_img = piv_img.unsqueeze(0)
    piv_emb1, qs = model1(piv_img)
    qs = qs / 2 - 0.3
    grad = torch.autograd.grad(qs, piv_img)
    grad = grad[0].numpy()
    g = np.reshape(grad, [3, grad.shape[2], grad.shape[3]])
    pgg = process_grad(g)
    plt.imshow(pgg, cmap=plt.get_cmap('RdYlGn'), vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig('fig2.png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/artorias/Downloads/MeGlass_120x120')
    parser.add_argument('--backbone', type=str, default='backbone.pth')
    parser.add_argument('--pose', type=str, default='pose.pth')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model1 = iresnet100()
    model1.load_state_dict(torch.load('weights/backbone.pth', map_location='cpu'))
    model1.eval()

    all_sub_folders = os.listdir(args.data)
    all_sub_folders = sorted(all_sub_folders)

    subfolder = all_sub_folders[0]
    image_path_list = glob.glob(os.path.join(args.data, '*.jpg'))
    image_path_list = sorted(image_path_list)
    get_fnmr(model1, pivot='7276470@N03_identity_3@13551953534_4.jpg', q_threshold=0.2)
