from backbones.model import Explainable_FIQA
from dataset.dataset import ExFIQA
from torch.utils.data import DataLoader
from tqdm import tqdm
from engine import Trainer
import torch.nn as nn
import pandas as pd
import torch
import argparse
from torchmetrics.classification import F1Score


def train_model(model, dataloader, dataset, optimizer, loss_fn, device, alpha):
    model.train()
    train_loss = 0.0
    count = 0

    for idx, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        count += 1
        optimizer.zero_grad()

        image = data[0].to(device)
        pose = data[1].to(device)
        br = data[2].to(device)
        pred_pose, pred_br = model(image)
        pose_loss = loss_fn(pred_pose, pose)
        br_loss = loss_fn(pred_br, br)
        loss = alpha * pose_loss + (1 - alpha) * br_loss
        train_loss += loss.item()
        loss.backward()

        optimizer.step()

    return train_loss / count


def valid_model(model, dataloader, dataset, loss_fn, device, f1, alpha):
    model.eval()
    train_loss = 0.0
    count = 0
    all_pose_preds = []
    all_pose_labels = []

    all_br_preds = []
    all_br_labels = []

    for idx, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        count += 1
        image = data[0].to(device)
        pose = data[1].to(device)
        br = data[2].to(device)
        pred_pose, pred_br = model(image)
        pose_loss = loss_fn(pred_pose, pose)
        br_poss = loss_fn(pred_br, br)
        loss = alpha * pose_loss + (1 - alpha) * br_poss
        train_loss += loss.item()
        ppose = pred_pose.max(1)[1]
        pbr = pred_br.max(1)[1]
        for i in range(len(data[1])):
            all_pose_labels.append(data[1][i])
            all_br_labels.append(data[2][i])
            all_pose_preds.append(ppose[i])
            all_br_preds.append(pbr[i])

    all_pose_labels = torch.tensor(all_pose_labels, dtype=int).resize(1, len(dataset)).squeeze(0)
    all_pose_preds = torch.tensor(all_pose_preds, dtype=int).resize(1, len(dataset)).squeeze(0)
    f1_pose = f1(all_pose_preds, all_pose_labels)

    all_br_labels = torch.tensor(all_br_labels, dtype=int).resize(1, len(dataset)).squeeze(0)
    all_br_preds = torch.tensor(all_br_preds, dtype=int).resize(1, len(dataset)).squeeze(0)
    f1_br = f1(all_br_preds, all_br_labels)

    return train_loss / count, f1_pose, f1_br


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--csv', type=str)
    parser.add_argument('--weight1', type=str)
    parser.add_argument('--weight2', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--load', type=bool)
    parser.add_argument('--alpha', type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Explainable_FIQA(train_q_only=True, weight_path=args.weight1)
    if args.load:
        model.load_state_dict(torch.load(args.weight2, map_location='cpu'), strict=False)
    model.to(device)

    train_val_dataframe = pd.read_csv(args.csv).iloc[:80186, :]
    train_df = train_val_dataframe.iloc[:60000, :]
    val_df = train_val_dataframe.iloc[60000:, :]

    train_dataset = ExFIQA(df=train_df)
    val_dataset = ExFIQA(df=val_df)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(lr=1e-3, params=model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(patience=1, factor=0.2, optimizer=opt)
    f1 = F1Score(task='multiclass', num_classes=2, average='macro')
    trainer = Trainer(lr_scheduler)

    for epoch in range(args.epochs):
        print(f'Epoch:{epoch}')
        trainloss = train_model(model, train_dataloader, train_dataset, opt, loss_fn, device, args.alpha)
        print(f'Train_loss:{trainloss}')
        val_loss, f1_pose, f1_br = valid_model(model, val_dataloader, val_dataset, loss_fn, device, f1, args.alpha)
        print(f'Valid_loss:{val_loss}')
        print(f'F1_pose:{f1_pose}')
        print(f'F1_brightness:{f1_br}')
        trainer(val_loss, model)
        if trainer.stop:
            break
