from dataset.dataset import ExFIQA
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import torch
import argparse
from torchmetrics.classification import F1Score
from backbones.model import PoseClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str,default='/home/artorias/PycharmProjects/weights/new_data.csv')
    parser.add_argument('--weight1', type=str, default='pose.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = PoseClassifier()
    model.load_state_dict(torch.load(args.weight1, map_location='cpu'))
    model.to(device)

    test_df = pd.read_csv(args.csv).iloc[102400:, :]


    test_dataset = ExFIQA(df=test_df)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    f1 = F1Score(task='multiclass', num_classes=7, average='none')

    print('Starting test...')
    all_preds = []
    all_labels = []


    model.eval()
    for idx, data in tqdm(enumerate(test_dataloader), total=int(len(test_dataset) / test_dataloader.batch_size)):
        image = data[0].to(device)
        pose = data[1].to(device)
        pred_pose = model(image)

        pred = pred_pose.max(1)[1]
        for i in range(len(data[1])):
            all_labels.append(data[1][i])
            all_preds.append(pred[i])

    all_labels = torch.tensor(all_labels, dtype=int).resize(1, len(test_dataset)).squeeze(0)
    all_preds = torch.tensor(all_preds, dtype=int).resize(1, len(test_dataset)).squeeze(0)
    f1_score = f1(all_preds, all_labels)
    print(f1_score)
