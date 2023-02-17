import pytorch_lightning as pl
from tqdm import tqdm
from torch.nn.functional import mse_loss


class MyCallBack(pl.Callback):
    def __init__(self, val_loader):
        super().__init__()
        self.val_loader = val_loader

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        sharp_mae = 0.0
        illu_mae = 0.0
        q_mae = 0.0
        count = 0
        for batch_idx, batch in tqdm(enumerate(self.val_loader)):
            count += 1
            image,qscore = batch

            qscore = qscore.to(pl_module._device)
            pred_q = pl_module(image)


            q_mae += mse_loss(qscore, pred_q, reduction='mean')

        q_mae /= count
        print('MAE for qscore {}'.format(float(q_mae)))


