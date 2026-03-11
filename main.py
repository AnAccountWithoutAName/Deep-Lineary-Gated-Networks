import lightning as L
import torch.nn as nn
from model import ControlConvNet
from data import ImageDataModule
import torch

def train():

    dm = ImageDataModule(
        block_size=3, grid_size=6, num_samples=1000,
        bg_mu=0, data_mu=5, tree_depth=4, 
        train_size=0.8, batch_size=32
    )
    

    model = ControlConvNet(
        in_channels=1, 
        hidden_channels=16, 
        num_layers=2,
        loss=nn.BCEWithLogitsLoss(),
        optim=torch.optim.Adam,
        lr=1e-3,
        weight_decay=1e-5
    )


    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto", 
        devices=1,
        log_every_n_steps=10
    )

 
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    train()