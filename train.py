from backbones.model import ExplainableFIQA
from dataset.dataset import ExFIQA
from torch.utils.data import DataLoader
from tqdm import tqdm
from engine import Trainer
import torch.nn as nn
import pandas as pd
import torch
import argparse
from torchmetrics.classification import F1Score


def train_model(model, dataloader, dataset, optimizer, loss_fn, device):
    model.train()
    train_loss = 0.0
    count = 0

    for idx, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        count += 1
        optimizer.zero_grad()
        image = data[0].to(device)
        pose = data[1].to(device)
        _,_,pred_pose = model(image)
        pose_loss = loss_fn(pred_pose, pose)

        train_loss += pose_loss.item()
        pose_loss.backward()
        optimizer.step()

    return train_loss / count


def valid_model(model, dataloader, dataset, loss_fn, device, f1):
    model.eval()
    train_loss = 0.0
    count = 0
    all_pose_preds = []
    all_pose_labels = []
    with torch.no_grad():
        for idx, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            count += 1
            image = data[0].to(device)
            pose = data[1].to(device)
            _,_,pred_pose = model(image)
            pose_loss = loss_fn(pred_pose, pose)

            train_loss += pose_loss.item()
            ppose = pred_pose.max(1)[1]
            for i in range(len(data[1])):
                all_pose_labels.append(data[1][i])
                all_pose_preds.append(ppose[i])

        all_pose_labels = torch.tensor(all_pose_labels, dtype=int).resize(1, len(dataset)).squeeze(0)
        all_pose_preds = torch.tensor(all_pose_preds, dtype=int).resize(1, len(dataset)).squeeze(0)
        f1_pose = f1(all_pose_preds, all_pose_labels)

        return train_loss / count, f1_pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--csv', type=str)
    parser.add_argument('--weight1', type=str)
    parser.add_argument('--weight2', type=str)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr',default=1e-2,type=float)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ExplainableFIQA(backbone_weight=args.weight1)
    model.to(device)

    train_val_dataframe = pd.read_csv(args.csv).iloc[:102400, :]
    train_df = train_val_dataframe.iloc[:93600, :]
    val_df = train_val_dataframe.iloc[93600:, :]

    train_dataset = ExFIQA(df=train_df)
    val_dataset = ExFIQA(df=val_df)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(lr=args.lr, params=model.pose_classifier.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(patience=1, factor=0.2, optimizer=opt)
    f1 = F1Score(task='multiclass', num_classes=7, average='none')
    trainer = Trainer(lr_scheduler)

    for epoch in range(args.epochs):
        print(f'Epoch:{epoch}')
        trainloss = train_model(model, train_dataloader, train_dataset, opt, loss_fn, device)
        print(f'Train_loss:{trainloss}')
        val_loss, f1_pose = valid_model(model, val_dataloader, val_dataset, loss_fn, device, f1)
        print(f'Valid_loss:{val_loss}')
        print(f'F1_pose: {f1_pose}')
        trainer(val_loss, model)
        if trainer.stop:
            break
