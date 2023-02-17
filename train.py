from backbones.model import Explainable_FIQA
from dataset.dataset import ExFIQA
from torch.utils.data import DataLoader
from callback.callback import MyCallBack
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
import torch.nn as nn
import pandas as pd
import torch


class EXFIQA(pl.LightningModule):
    def __init__(self, model, train_loader, val_loader, device):
        super().__init__()
        self.model = model
        self._device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mse = nn.MSELoss()

    def forward(self, image):
        image = image.to(self._device)
        return self.model(image)

    def configure_optimizers(self):
        optim1 = torch.optim.Adam(self.model.quality.parameters(), lr=1e-2)


        return optim1

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_idx):

        image,qscore = batch
        qscore = qscore.to(self._device)
        pred_qscore = self.forward(image)
        loss3 = self.mse(qscore, pred_qscore)
        return {'loss': loss3}

    def validation_step(self, batch, batch_idx):
        pass

if __name__ == '__main__':
    dataframe = pd.read_csv('/kaggle/input/train-q-only/MYFIQAC/data.csv')

    # test_len = int(len(dataframe) * 0.2)
    train_dataframe = dataframe.iloc[:64148, :]
    valid_dataframe = dataframe.iloc[64148:80185, :]


    model = Explainable_FIQA()
    train_dataset = ExFIQA(df=train_dataframe)
    valid_dataset = ExFIQA(df=valid_dataframe)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=8)

    module = EXFIQA(model=model, train_loader=train_loader, val_loader=val_loader, device=torch.device('cuda'))
    callback = MyCallBack(val_loader)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(max_epochs=20, callbacks=[callback, lr_monitor], auto_lr_find=True, accelerator='gpu')

    trainer.fit(module)
    torch.save(model.state_dict(), 'model.pth')
