import torch 
import torch.nn.functional as F
import torch_geometric 
from lightning import LightningModule

from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression.mse import MeanSquaredError
from torchmetrics.regression.mae import MeanAbsoluteError

from net.model import WavePE_AutoEncoder


class Lightning_WavePE_AutoEncoder(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.net = WavePE_AutoEncoder(config)
        self.weight = torch.tensor([config.pos_weight]).to(config.device)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_best_loss = MinMetric()

    def forward(self, data):
        return self.net(data)

    def on_train_start(self):
        self.val_loss.reset()

    def model_step(self, data):
        A_hat, A, mask = self.net.fit(data)
        A_hat = A_hat.squeeze()
        loss = F.binary_cross_entropy(A_hat, A, reduction = 'none', weight=self.weight)
        loss = mask.unsqueeze(1) * loss * mask.unsqueeze(-1)
        loss = loss.mean()
        return loss

    def training_step(self, data, idx):
        loss = self.model_step(data)

        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss

    def validation_step(self, data, idx):
        loss = self.model_step(data)

        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss
         
    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()
        self.val_best_loss(loss)
        self.log("val_best_loss", self.val_best_loss.compute(), sync_dist = True, prog_bar = True)

    def configure_optimizers(self):
        optimizer =torch.optim.Adam(params = self.net.parameters(), lr = self.hparams.config.lr)
        return {"optimizer": optimizer}