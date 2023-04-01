from backbones.model import ExplainableFIQA
from backbones.iresnet import iresnet100
from dataset.dataset import ExFIQA
from torch.utils.data import DataLoader
import pandas as pd
import torch
import argparse
from torchmetrics.classification import F1Score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--csv', type=str)
    parser.add_argument('--weight1', type=str)
    parser.add_argument('--weight2', type=str,default='weights/backbone.pth')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', default=1e-2, type=float)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = iresnet100()
    model.load_state_dict(torch.load(args.weight1, map_location='cpu'),strict=False)
    #model.backbone.load_state_dict(torch.load(args.weight2,map_location='cpu'))
    model.to(device)
    model.eval()
    train_val_dataframe = pd.read_csv(args.csv).iloc[:102400, :]
    train_df = train_val_dataframe.iloc[:100, :]
    val_df = train_val_dataframe.iloc[93600:, :]
    test_df = pd.read_csv(args.csv).iloc[102400:102500, :]

    train_dataset = ExFIQA(df=train_df)
    val_dataset = ExFIQA(df=val_df)
    test_dataset = ExFIQA(df=test_df)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():

        for data in train_dataloader:
            image = data[0].to(device)
            pose = data[1].to(device)
            _, qs = model(image)
            print(qs/2-0.2)




