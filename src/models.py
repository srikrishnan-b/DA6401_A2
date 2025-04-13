# Loading the required libraries

import torch
import wandb
from torch import nn
import lightning as pl
from torchmetrics import Accuracy
from lightning.pytorch.loggers import WandbLogger
import torchvision.models as models
from torchvision.models import (
    googlenet,
    GoogLeNet_Weights,
    resnet50,
    vgg11,
    inception_v3,
    efficientnet_v2_s,
    vit_b_16,
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
    def __init__(self, model: str, finetune_strat: str, optim: str, n_classes, lr):
        super().__init__()
        self.optim = optim
        self.save_hyperparameters()
        self.model = load_torch_models(model, finetune_strat, n_classes)
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
        if self.optim == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.lr, momentum=0.9
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


def load_torch_models(model, finetune_strat, n_classes):
    # TODO finetune strategies
    if model == "googlenet":
        googlenet_model = googlenet(
            weights=GoogLeNet_Weights.IMAGENET1K_V1, aux_logits=False
        )
        for param in googlenet_model.parameters():
            param.requires_grad = False
        for param in googlenet_model.fc.parameters():
            param.requires_grad = True

        googlenet_model.fc = nn.Linear(
            in_features=googlenet_model.fc.in_features, out_features=n_classes
        )

        return googlenet_model

    elif model == "resnet50":
        resnet_model = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet_model.parameters():
            param.requires_grad = False
        for param in resnet_model.fc.parameters():
            param.requires_grad = True
        resnet_model.fc = nn.Linear(
            in_features=resnet_model.fc.in_features, out_features=n_classes
        )

        return resnet_model

    elif model == "vgg11":
        vgg_model = vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
        for param in vgg_model.parameters():
            param.requires_grad = False
        for param in vgg_model.classifier.parameters():
            param.requires_grad = True
        vgg_model.classifier[6] = nn.Linear(
            in_features=vgg_model.classifier[6].in_features, out_features=n_classes
        )

        return vgg_model

    elif model == "inception":
        inception_model = inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True
        )
        for param in inception_model.parameters():
            param.requires_grad = False
        for param in inception_model.fc.parameters():
            param.requires_grad = True
        inception_model.fc = nn.Linear(
            in_features=inception_model.fc.in_features, out_features=n_classes
        )

        return inception_model

    elif model == "efficient":
        efficientnet_model = efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        for param in efficientnet_model.parameters():
            param.requires_grad = False
        for param in efficientnet_model.fc.parameters():
            param.requires_grad = True
        efficientnet_model.classifier[1] = nn.Linear(
            in_features=efficientnet_model.classifier[1].in_features,
            out_features=n_classes,
        )

        return efficientnet_model

    elif model == "visiontransformer":
        vit_model = vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        for param in vit_model.parameters():
            param.requires_grad = False
        for param in vit_model.fc.parameters():
            param.requires_grad = True
        vit_model.head[0] = nn.Linear(
            in_features=vit_model.head[0].in_features, out_features=n_classes
        )
        return vit_model
