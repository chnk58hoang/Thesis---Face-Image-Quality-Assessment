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


def train_model(model, dataloader, dataset, optimizers, loss_fn, device):
    model.train()
    train_loss = 0.0
    count = 0

    for idx, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        count += 1
        optimizers.zero_grad()
        image = data[0].to(device)
        qscore = data[1].to(device)
        pred_qscore = model(image)
        loss = loss_fn(pred_qscore,qscore)
        train_loss += loss.item()

        loss.backward()
        optimizers.step()

    return train_loss / count


def valid_model(model, dataloader, dataset, loss_fn, device, f1):
    model.eval()
    train_loss = 0.0
    count = 0

    all_preds = []
    all_labels = []

    for idx, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        count += 1
        image = data[0].to(device)
        pose = data[1].to(device)
        pred_pose = model(image)
        loss = loss_fn(pred_pose,pose)
        train_loss += loss.item()
        pred = pred_pose.max(1)[1]
        for i in range(len(data[1])):
            all_labels.append(data[1][i])
            all_preds.append(pred[i])

    all_labels = torch.tensor(all_labels, dtype=int).resize(1, len(dataset)).squeeze(0)
    all_preds = torch.tensor(all_preds, dtype=int).resize(1, len(dataset)).squeeze(0)

    f1_score = f1(all_preds, all_labels)

    return train_loss / count, f1_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--csv', type=str)
    parser.add_argument('--weight1', type=str)
    parser.add_argument('--weight2', type=str)
    parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Explainable_FIQA(train_q_only=True, weight_path=args.weight1)
    #model.load_state_dict(torch.load(args.weight2, map_location='cpu'), strict=False)
    #model.to(device)

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
    f1 = F1Score(task='multiclass', num_classes=2)

    trainer = Trainer(lr_scheduler)

    for epoch in range(args.epochs):
        print(f'Epoch:{epoch}')
        trainloss = train_model(model, train_dataloader, train_dataset, opt, loss_fn, device)
        print(f'Train_loss:{trainloss}')
        val_loss, f1_score = valid_model(model, val_dataloader, val_dataset, loss_fn, device, f1)
        print(f'Valid_loss:{val_loss}')
        print(f'F1:{f1_score}')
        trainer(val_loss, model)
        if trainer.stop:
            break
