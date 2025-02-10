from pathlib import Path

import yaml
import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam, SGD, AdamW, RMSprop
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
import torch.nn.functional as F

from histolung.mil.utils import get_optimizer, get_scheduler, get_loss_function


class AggregatorPL(pl.LightningModule):

    def __init__(self,
                 input_dim,
                 hidden_dim=None,
                 num_classes=2,
                 feature_extractor=None):
        super(AggregatorPL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.feature_extractor = feature_extractor

    def forward(self, x):
        raise NotImplementedError(
            "This method should be implemented by subclasses.")

    def load_feature_extractor(self):
        raise NotImplementedError(
            "This method should be implemented by subclasses.")

    @staticmethod
    def from_config(config):
        """
        Instantiate an aggregator class based on the YAML configuration.

        Args:
            config (dict): Parsed configuration dictionary containing the aggregator settings.

        Returns:
            AggregatorPL: An instance of the specified aggregator class.
        """
        if isinstance(config, (Path, str)):
            with open(config, "r") as file:
                config = yaml.safe_load(file)

        # Registry of supported aggregators
        aggregator_registry = {
            "attention": AttentionAggregatorPL,
            "mean": MeanPoolingAggregatorPL,
        }

        # Extract aggregator settings
        aggregator_name = config["aggregator"]["name"]
        aggregator_cls = aggregator_registry.get(aggregator_name)
        if aggregator_cls is None:
            raise ValueError(f"Unsupported aggregator type: {aggregator_name}")

        # Extract common arguments
        kwargs = config["aggregator"].get("kwargs", {})
        # kwargs["input_dim"] = kwargs.get("input_dim")
        # kwargs["hidden_dim"] = kwargs.get("hidden_dim")
        kwargs["num_classes"] = kwargs.get("num_classes", 2)

        # Include feature extractor metadata
        feature_extractor = config.get("feature_extractor", {})
        kwargs["feature_extractor"] = feature_extractor

        # Training-related parameters
        training_params = {
            "optimizer": config["training"]["optimizer"],
            "optimizer_kwargs": config["training"]["optimizer_kwargs"],
            "scheduler": config["training"]["lr_scheduler"],
            "scheduler_kwargs": config["training"]["lr_scheduler_kwargs"],
            "loss": config["training"]["loss"],
            "loss_kwargs": config["training"]["loss_kwargs"],
        }

        # Pass common arguments and training parameters
        return aggregator_cls(**kwargs, **training_params)


class AttentionAggregatorPL(AggregatorPL):

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_classes=2,
        dropout=0.2,
        optimizer="adam",
        optimizer_kwargs=None,
        scheduler=None,
        scheduler_kwargs=None,
        loss="BCEWithLogitsLoss",
        loss_kwargs=None,
        feature_extractor=None,
    ):
        super(AttentionAggregatorPL, self).__init__(
            input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            feature_extractor=feature_extractor,
        )
        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.scheduler_name = scheduler
        self.scheduler_kwargs = scheduler_kwargs

        # Define layers
        self.projection_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(p=dropout),
        )
        self.pre_fc_layer = nn.Sequential(
            nn.Linear(hidden_dim * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=0),
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        if loss_kwargs:
            self.loss_fn = get_loss_function(loss, **loss_kwargs)
        else:
            self.loss_fn = get_loss_function(loss)

        # Metrics:
        self.train_accuracy = Accuracy(
            num_classes=num_classes,
            average='macro',
            task="multiclass",
        )
        self.val_accuracy = Accuracy(
            num_classes=num_classes,
            average='macro',
            task="multiclass",
        )

    def forward(self, x):
        x = self.projection_layer(x) # (num_patches, hidden_dim)
        attention = self.attention(x) # (num_patches, num_classes)
        attention = torch.transpose(attention, 1, 0) # (num_classes, num_patches)
        aggregated_embedding = torch.mm(attention, x) # (num_classes, hidden_dim)
        aggregated_embedding = aggregated_embedding.view(
            -1,
            self.hidden_dim * self.num_classes,
        ) # (num_patches, hidden_dim)
        output = self.pre_fc_layer(aggregated_embedding) # (1, hidden_dim)
        output = self.fc(output) # (1, num_classes)
        return torch.squeeze(output), attention

    def training_step(self, batch, batch_idx):
        _, embeddings, labels = batch
        # labels = labels.to(self.device)
        batch_outputs = []
        batch_size = len(labels)

        for embedding in embeddings:
            outputs, _ = self(embedding)
            batch_outputs.append(outputs)

        batch_outputs = torch.stack(batch_outputs)
        labels_one_hot = F.one_hot(labels,
                                   num_classes=self.num_classes).float()
        loss = self.loss_fn(batch_outputs, labels_one_hot)

        # Compute accuracy
        preds = torch.argmax(batch_outputs, dim=-1)
        self.train_accuracy(preds, labels)

        # Log metrics
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 batch_size=batch_size,
                 prog_bar=True)
        self.log('train_acc',
                 self.train_accuracy,
                 on_step=True,
                 on_epoch=True,
                 batch_size=batch_size,
                 prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, embeddings, labels = batch
        # labels = labels.to(self.device)
        batch_outputs = []
        batch_size = len(labels)

        for embedding in embeddings:
            output, _ = self(embedding)
            batch_outputs.append(output)

        batch_outputs = torch.stack(batch_outputs)
        labels_one_hot = F.one_hot(labels,
                                   num_classes=self.num_classes).float()
        loss = self.loss_fn(batch_outputs, labels_one_hot)

        # Compute accuracy
        preds = torch.argmax(batch_outputs, dim=-1)
        self.val_accuracy(batch_outputs, labels)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc',
                 self.val_accuracy,
                 on_step=False,
                 on_epoch=True,
                 batch_size=batch_size,
                 prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds, _ = self(x)
        loss = self.loss_fn(preds, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):

        optimizer = get_optimizer(
            self.parameters(),
            self.optimizer_name,
            **self.optimizer_kwargs,
        )
        scheduler = get_scheduler(
            optimizer,
            self.scheduler_name,
            **self.scheduler_kwargs,
        )

        if isinstance(scheduler, dict):
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return [optimizer], [scheduler]


class MeanPoolingAggregatorPL(AggregatorPL):

    def __init__(self, input_dim, num_classes=2):
        super(MeanPoolingAggregatorPL, self).__init__(input_dim,
                                                      num_classes=num_classes)

    def forward(self, x):
        # Compute the mean embedding
        aggregated_embedding = torch.mean(x, dim=0)
        return aggregated_embedding, None
