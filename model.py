import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import BinaryAccuracy

class DLGN_Conv_1(L.LightningModule):
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, beta, 
                 num_layers, criterion, optimizer, lr, weight_decay, padding='same'):
        super().__init__()
        self.save_hyperparameters()
        
        self.beta = beta
        self.criterion = criterion
        self.lr = lr
        self.optim_class = optimizer
        self.weight_decay = weight_decay
        self.num_layers = num_layers

        self.gating_layers = nn.ModuleList()
        self.value_layers = nn.ModuleList()
        

        self.value_layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding='same', bias=True))
        for _ in range(num_layers):
            self.value_layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size = 3, padding = 'same', bias = True))
        self.value_layers.append(nn.Conv2d(hidden_channels,))

       
        
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

    def forward(self, x):
        
        v = self.value_layers[0](x)
        
        
        for i in range(self.num_layers):
            g = self.gating_layers[i](x)
            gate_mask = torch.sigmoid(self.beta * g)
       
            v = self.value_layers[i+1](v)
            v = v * gate_mask
  
        h_L = F.adaptive_avg_pool2d(v, (1, 1))
        h_L = torch.flatten(h_L, 1)

 
        logits = h_L @ self.u_L_plus_1
        
        return logits

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        preds = (torch.sigmoid(logits) > 0.5).float()
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return self.optim_class(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)



class ControlConvNet(L.LightningModule):
    def __init__(self, in_channels, hidden_channels, num_layers, loss, optim, lr, weight_decay):
        super().__init__() # Required for LightningModule
        self.save_hyperparameters() # Handy for logging/reloading
        
        self.in_channels = in_channels
        self.num_layers = num_layers

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Flatten(),
            nn.LazyLinear(out_features=1),
        )
        
        # Unifying the loss function name
        self.criterion = loss 
        self.optim_class = optim
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

    def forward(self, x):
        return self.layers(x).squeeze(1) # Squeeze to match shape (batch_size,)

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        
        # Convert logits to probabilities/preds for accuracy
        # Assuming Binary Cross Entropy with Logits
        preds = (torch.sigmoid(logits) > 0.5).long()
        
        return loss, logits, y, preds

    def training_step(self, batch, batch_idx):
        loss, logits, y, preds = self._shared_step(batch)
        
        # Log loss and update accuracy
        self.train_acc(preds, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y, preds = self._shared_step(batch)
        
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = self.optim_class(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        return optimizer