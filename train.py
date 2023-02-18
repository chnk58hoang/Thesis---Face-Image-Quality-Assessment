from backbones.model import Explainable_FIQA
from dataset.dataset import ExFIQA
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import torch
import argparse


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
        loss = loss_fn(qscore, pred_qscore)
        train_loss += loss.item()

        loss.backward()
        optimizers.step()

    return train_loss / count


def valid_model(model, dataloader, dataset, loss_fn, device):
    model.eval()
    train_loss = 0.0
    count = 0

    for idx, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        count += 1
        image = data[0].to(device)
        qscore = data[1].to(device)
        pred_qscore = model(image)
        loss = loss_fn(qscore, pred_qscore)
        train_loss += loss.item()

    return train_loss / count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--csv', type=str)
    parser.add_argument('--weight', type=str)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Explainable_FIQA(train_q_only=True,weight_path=args.weight).to(device)

    train_val_dataframe = pd.read_csv(args.csv).iloc[:80186, :]
    train_df = train_val_dataframe.iloc[:60000, :]
    val_df = train_val_dataframe.iloc[60000:, :]

    train_dataset = ExFIQA(df=train_df)
    val_dataset = ExFIQA(df=val_df)

    train_dataloader = DataLoader(train_dataset, batch_size=32)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    loss_fn = nn.L1Loss()
    opt = torch.optim.Adam(lr=1e-3, params=model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(patience=1, factor=0.5, optimizer=opt)

    for epoch in range(args.epochs):
        print(f'Epoch:{epoch}')
        trainloss = train_model(model, train_dataloader, train_dataset, opt, loss_fn, device)
        print(f'Train_loss:{trainloss}')
        val_loss = valid_model(model, val_dataloader, val_dataset, loss_fn, device)
        print(f'Valid_loss:{val_loss}')
        lr_scheduler.step(val_loss)

    torch.save(model.state_dict(),'model.pth')