# Loading the required libraries

import torch
import wandb
from torch import nn
import lightning as pl
from torchmetrics import Accuracy
from lightning.pytorch.loggers import WandbLogger
import torchvision.models as models
from lightning.pytorch.callbacks import BaseFinetuning
from torchvision.models import (
    
    resnet50,
    vgg11,
)


# CNN model
class CNN(nn.Module):
    def __init__(
        self,
        input,
        filters,
        kernel,
        pool_kernel,
        pool_stride,
        batchnorm,
        activation,
        dropout,
        ffn_size,
        num_classes=10,
    ):
        super().__init__()

        self.act = self._activation(activation)
        self.convblock1 = self._convblock(
            input,
            filters[0],
            kernel[0],
            pool_kernel[0],
            pool_stride[0],
            self.act,
            batchnorm,
            dropout,
        )
        self.convblock2 = self._convblock(
            filters[0],
            filters[1],
            kernel[1],
            pool_kernel[1],
            pool_stride[1],
            self.act,
            batchnorm,
            dropout,
        )
        self.convblock3 = self._convblock(
            filters[1],
            filters[2],
            kernel[2],
            pool_kernel[2],
            pool_stride[2],
            self.act,
            batchnorm,
            dropout,
        )
        self.convblock4 = self._convblock(
            filters[2],
            filters[3],
            kernel[3],
            pool_kernel[3],
            pool_stride[3],
            self.act,
            batchnorm,
            dropout,
        )
        self.convblock5 = self._convblock(
            filters[3],
            filters[4],
            kernel[4],
            pool_kernel[4],
            pool_stride[4],
            self.act,
            batchnorm,
            dropout,
        )
        if batchnorm:
            self.batch_norm = nn.BatchNorm1d(num_features=ffn_size)
        else:
            self.batch_norm = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(ffn_size)
        self.out = nn.Linear(ffn_size, num_classes)

    def _convblock(
        self,
        input,
        output,
        kernel,
        pool_kernel,
        pool_stride,
        activation_fn,
        batchnorm,
        dropout,
    ):

        if batchnorm:
            return nn.Sequential(
                nn.Conv2d(input, output, kernel),
                activation_fn,
                nn.BatchNorm2d(output),
                # nn.Dropout(dropout),
                nn.MaxPool2d(pool_kernel, pool_stride),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input, output, kernel),
                activation_fn,
                # nn.Dropout(dropout),
                nn.MaxPool2d(pool_kernel, pool_stride),
            )

    def _activation(self, act):
        if act == "relu":
            act = nn.ReLU()
        elif act == "gelu":
            act = nn.GELU()
        elif act == "selu":
            act = nn.SELU()
        elif act == "mish":
            act = nn.Mish()
        elif act == "swish":
            act = nn.SiLU()
        return act

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.out(x)
        return x


# Pytorch lighnting implementation of model
class CNN_light(pl.LightningModule):
    def __init__(
        self,
        optim,
        filters,
        kernel,
        pool_kernel,
        pool_stride,
        batchnorm,
        activation,
        dropout,
        ffn_size,
        n_classes,
        lr,
    ):
        super().__init__()
        self.optim = optim
        self.save_hyperparameters()
        self.model = CNN(
            input=3,
            filters=filters,
            kernel=kernel,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            batchnorm=batchnorm,
            activation=activation,
            dropout=dropout,
            ffn_size=ffn_size,
            num_classes=n_classes,
        )
        self.train_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.train_accuracy(logits, y)
        self.log("train loss", loss, on_step=False, on_epoch=True)
        self.log("train accuracy", acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.val_accuracy(logits, y)
        self.log("val loss", loss, on_step=False, on_epoch=True)
        self.log("val accuracy", acc, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.val_accuracy(logits, y)
        self.log("test loss", loss, on_step=False, on_epoch=True)
        self.log("test accuracy", acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        if self.optim == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.lr, momentum=0.9
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


# Pytorch lighnting implementation of model with pretrained weights
class pretrained_light(pl.LightningModule):
    def __init__(self, model: str, optim: str, n_classes, lr):
        super().__init__()
        self.optim = optim
        self.ptmodel= model
        self.save_hyperparameters()
        self.model = load_torch_models(model, n_classes)
        self.lr = lr
        self.train_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = self.loss_fn(logits, y)
        acc = self.train_accuracy(logits, y)
        self.log("train loss", loss, on_step=False, on_epoch=True)
        self.log("train accuracy", acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.val_accuracy(logits, y)
        self.log("val loss", loss, on_step=False, on_epoch=True)
        self.log("val accuracy", acc, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.val_accuracy(logits, y)
        self.log("test loss", loss, on_step=False, on_epoch=True)
        self.log("test accuracy", acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
        return optimizer

class Unfreeze_after_epochs(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self._to_freeze = None

    def freeze_before_training(self, pl_module):
      to_freeze = [pl_module.model.conv1,
                   pl_module.model.bn1,
                   pl_module.model.layer1, 
                   pl_module.model.layer2,
                   pl_module.model.layer3             
                   ]
      self._to_freeze = to_freeze
      self.freeze(modules = to_freeze)
      self.make_trainable(pl_module.model.fc)

    def finetune_function(self, pl_module, current_epoch, optimizer):
      if current_epoch == self._unfreeze_at_epoch:
        self.unfreeze_and_add_param_group(modules = self._to_freeze,
                                          optimizer=optimizer, train_bn=True, lr=pl_module.lr*0.1)





def load_torch_models(model, n_classes):

    if model == "resnet50":

        model = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model


    elif model=='vgg11':
        model = vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)
        return model



    


