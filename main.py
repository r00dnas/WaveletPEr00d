import os 
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric

from tqdm import tqdm
from easydict import EasyDict
import yaml

from data import Lightning_Dataset
from net.lightning_model import Lightning_WavePE_AutoEncoder

from lightning import Trainer, seed_everything
from lightning.pytorch import loggers, callbacks

import warnings 
warnings.filterwarnings("ignore")

with open("config/pretrain.yml", "r") as f:
    dict_config = yaml.safe_load(f)
    config = EasyDict(dict_config)

## fix random seed for reproducibility ##
seed_everything(config.seed, workers = True)
torch.set_float32_matmul_precision(config.precision)

datamodule = Lightning_Dataset(config)
model = Lightning_WavePE_AutoEncoder(config)
#model = Lightning_WavePE_AutoEncoder.load_from_checkpoint(config.ckpt_path)


#### set up ####
if config.debug:
    logger_list = []
else:
    logger_list = [
        loggers.WandbLogger(
            save_dir = config.output_dir, 
            project = config.wandb_project, 
            config= dict_config, 
            log_model = False, 
        )
    ]

ckpt_name = f"{config.data_name}_" + "{epoch:02d}_{train_loss:.3f}_{val_loss:.3f}_{val_best_loss:.3f}"
ckpt_dirpath = os.path.join(config.output_dir, f"ckpts/{config.data_name}_{config.version}")
os.makedirs(ckpt_dirpath, exist_ok=True)

callback_list = [
    callbacks.RichModelSummary(),
    callbacks.RichProgressBar(),
    callbacks.ModelCheckpoint(
        dirpath = ckpt_dirpath, 
        filename = ckpt_name, 
        monitor = "val_best_loss", 
        verbose = True, 
        save_top_k = config.num_ckpts, 
        save_weights_only = False, 
        save_last = False, 
        every_n_epochs = config.save_every_n_epochs
    )
]


trainer = Trainer(accelerator="cpu", devices=1, max_epochs=config.num_epoch, default_root_dir=config.output_dir, logger=logger_list, callbacks=callback_list)


trainer.fit(model, datamodule, ckpt_path = config.ckpt_path)





