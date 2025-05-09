"""This file contains codes for pytorch lighting implementation of CNN model, Finetuning model, and training routine for sweep"""


# Loading the required libraries
import torch
import wandb
from torch import nn
import lightning as pl
from torchmetrics import Accuracy
from lightning.pytorch.loggers import WandbLogger
import torchvision.models as models
from lightning.pytorch.callbacks import BaseFinetuning
from torchvision.models import resnet50
from src.dataloader import iNat_dataset
from src.config import *
from src.sweep_config import *
import gc


#==============================================================================================================
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
            
        )
        self.convblock2 = self._convblock(
            filters[0],
            filters[1],
            kernel[1],
            pool_kernel[1],
            pool_stride[1],
            self.act,
            batchnorm,
        )
        self.convblock3 = self._convblock(
            filters[1],
            filters[2],
            kernel[2],
            pool_kernel[2],
            pool_stride[2],
            self.act,
            batchnorm,
        )
        self.convblock4 = self._convblock(
            filters[2],
            filters[3],
            kernel[3],
            pool_kernel[3],
            pool_stride[3],
            self.act,
            batchnorm,
        )
        self.convblock5 = self._convblock(
            filters[3],
            filters[4],
            kernel[4],
            pool_kernel[4],
            pool_stride[4],
            self.act,
            batchnorm,
        )
        if batchnorm:
            self.batch_norm = nn.BatchNorm1d(num_features=ffn_size)
        else:
            self.batch_norm = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(ffn_size)
        self.out = nn.Linear(ffn_size, num_classes)


    ## Convolutional block is defined with conv layer, batchnorm, activation and maxpooling 
    def _convblock(
        self,
        input,
        output,
        kernel,
        pool_kernel,
        pool_stride,
        activation_fn,
        batchnorm,
        
    ):

        if batchnorm:
            return nn.Sequential(
                nn.Conv2d(input, output, kernel),
                activation_fn,
                nn.BatchNorm2d(output),
                nn.MaxPool2d(pool_kernel, pool_stride),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input, output, kernel),
                activation_fn,
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

# ===============================================================================================================


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

# ================================================================================================================

# Function to train the model
def trainCNN(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config

        run.name = f"A_{config.augmentation}_D_{config.dropout:.2f}_bn_{config.batchnorm}_ffn_{config.ffn_size}"
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            # Getting train, validation, and test dataloaders
            dataset = iNat_dataset(
                data_dir=data_dir,
                augmentation=aug,
                batch_size=batch_size,
                NUM_WORKERS=NUM_WORKERS,
            )
            train_dataloader, val_dataloader, _, classes, n_classes = (
                dataset.load_dataset()
            )
            # Model
            model = CNN_light(
                optim=config.optim,
                filters=config.filters,
                kernel=config.kernel,
                pool_kernel=config.pool_kernel,
                pool_stride=config.pool_stride,
                batchnorm=config.batchnorm,
                activation=config.activation,
                dropout=config.dropout,
                ffn_size=config.ffn_size,
                n_classes=n_classes,
                lr=config.lr,
            )
            logger = WandbLogger(
                project=project_name, name=run.name, experiment=run, log_model=False
            )
            trainer = pl.Trainer(
                devices=1,
                accelerator="auto",
                precision="16-mixed",
                gradient_clip_val=1.0,
                max_epochs=config.epochs,
                logger=logger,
                profiler=None,
            )

            trainer.fit(model, train_dataloader, val_dataloader)
        finally:
            del trainer
            del model
            gc.collect()
            torch.cuda.empty_cache()

# ===============================================================================================================




# Pytorch lighnting implementation of model with pretrained weights
class pretrained_light(pl.LightningModule):
    def __init__(self, model: str, optim: str, n_classes, lr):
        super().__init__()
        self.optim = optim
        self.save_hyperparameters()
        self.model = load_torch_model(n_classes)                   # Load resnet50
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
        # filter parameters that are frozen
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters())
        )
        return optimizer

# ================================================================================================================

# Callback to freeze and unfreeze layers in the pretrained model
class Unfreeze_after_epochs(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=[5]):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self._to_freeze = None

    def freeze_before_training(self, pl_module):
        # Freeze all layers except the classifier 
        to_freeze = [
            pl_module.model.conv1,
            pl_module.model.bn1,
            pl_module.model.layer1,
            pl_module.model.layer2,
            pl_module.model.layer3,
        ]
        self._to_freeze = to_freeze
        self.freeze(modules=to_freeze)
        self.make_trainable(pl_module.model.fc)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # Unfreeze layers at specified epochs
        if current_epoch == self._unfreeze_at_epoch[0]:
            print("Unfreezing layer 3,2")
            self.unfreeze_and_add_param_group(
                modules=self._to_freeze[-2:],
                optimizer=optimizer,
                train_bn=True,
                lr=pl_module.lr * 0.1,
            )


#================================================================================================================
# Function to load the pretrained model
def load_torch_model(n_classes):

    model = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Changing the final layer
    model.fc = nn.Sequential(
        nn.Linear(in_features=model.fc.in_features, out_features=1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, n_classes),
    )

    return model

#================================================================================================================