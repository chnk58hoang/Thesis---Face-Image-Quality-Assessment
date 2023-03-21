from matplotlib import pyplot as plt
from torchvision import transforms
from torch.nn.functional import cosine_similarity
import torch
import argparse
from backbones.iresnet import iresnet50,iresnet100
import numpy as np
import cv2
import glob
from tqdm import tqdm
import os
from PIL import Image

def get_fnmr(model, all_imgs, q_threshold,s_threshold=0.3):
    rej = 0
    fnm = 0
    pivot = all_imgs.pop(64)
    piv_img = cv2.imread(pivot)
    piv_img = cv2.resize(piv_img, (112, 112))
    piv_img = np.transpose(piv_img, (2, 0, 1))
    piv_img = np.expand_dims(piv_img,axis=0)
    piv_img = torch.Tensor(piv_img)
    piv_img.div_(255).sub_(0.5).div_(0.5)
    piv_emb, _,_ = model(piv_img)

    for img in all_imgs:
        image = cv2.imread(img)
        image = cv2.resize(image, (112, 112))
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.Tensor(image)
        image.div_(255).sub_(0.5).div_(0.5)
        emb, qscore,_ = model(image)
        print(img)
        print(qscore / 2 - 0.2)
        cs = cosine_similarity(emb, piv_emb)
        if qscore < q_threshold:

            rej += 1
        elif cs < s_threshold:
            fnm += 1

    return rej, fnm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/artorias/Downloads/multi_PIE_crop_128')
    parser.add_argument('--weight1', type=str, default='181952backbone.pth')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = iresnet100(use_se = False)
    model.load_state_dict(torch.load(args.weight1, map_location='cpu'),strict=False)
    model.to(device)
    model.eval()

    all_sub_folders = os.listdir(args.data)
    all_sub_folders = sorted(all_sub_folders)

    for subfolder in tqdm(all_sub_folders[200:]):
        image_path_list = glob.glob(os.path.join(args.data,subfolder,'*.png'))
        image_path_list = sorted(image_path_list)
        rej,fnm = get_fnmr(model,image_path_list,q_threshold=0.2)
        print(rej,fnm)


