import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW, RMSprop


class MILLightningModule(pl.LightningModule):

    def __init__(self,
                 model,
                 loss_fn,
                 optimizer_config,
                 num_classes,
                 hdf5_file=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config
        self.num_classes = num_classes
        self.hdf5_file = hdf5_file  # For embedding training

    def forward(self, embeddings):
        return self.model(embeddings)

    def training_step(self, batch, batch_idx):
        wsi_ids, labels = batch
        labels = labels.to(self.device)
        batch_outputs = []

        for wsi_id in wsi_ids:
            embeddings = self.get_embeddings(wsi_id)
            outputs, _ = self(embeddings)
            batch_outputs.append(outputs)

        batch_outputs = torch.stack(batch_outputs)
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes)
        loss = self.loss_fn(batch_outputs, labels_one_hot.float())

        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        wsi_ids, labels = batch
        labels = labels.to(self.device)
        batch_outputs = []

        for wsi_id in wsi_ids:
            embeddings = self.get_embeddings(wsi_id)
            outputs, _ = self(embeddings)
            batch_outputs.append(outputs)

        batch_outputs = torch.stack(batch_outputs)
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes)
        loss = self.loss_fn(batch_outputs, labels_one_hot.float())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def get_embeddings(self, wsi_id):
        embeddings = torch.tensor(self.hdf5_file['embeddings'][wsi_id][:]).to(
            self.device)
        return embeddings

    def configure_optimizers(self, optimizer_cfg):
        optimizer_dict = {
            "Adam": Adam,
            "AdamW": AdamW,
            "SGD": SGD,
            "RMSprop": RMSprop
        }
        optimizer_name = optimizer_cfg["optimizer"]

        optimizer_class = optimizer_dict.get(optimizer_name)

        if optimizer_class is None:
            raise ValueError(f"Optimizer '{optimizer_name}' not supported")

        optimizer = torch.optim.Adam(self.parameters(),
                                     **self.optimizer_config)
        return optimizer
