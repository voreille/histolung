import logging
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from histolung.models.models import MILModel
from histolung.mil.data_loader import TileDataset
from histolung.mil.data_augmentation import get_augmentations_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class MILTrainer:

    def __init__(
        self,
        model: MILModel,
        dataloaders: Dict[str, DataLoader],
        optimizer: Optimizer,
        loss_fn,
        tile_paths_by_wsi=None,
        device='cuda',
        tile_preprocess=None,
        tile_augmentation=None,
    ):
        """
        Initializes the MILTrainer class.
        
        Args:
            model (MILModel): MIL model to be trained.
            dataloaders (dict): A dictionary containing 'train' and 'val' DataLoaders.
            optimizer (Optimizer): Optimizer for model training.
            loss_fn: Loss function used for training (e.g., CrossEntropyLoss).
            device (str): Device to train the model on ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.tile_preprocess = tile_preprocess
        self.tile_augmentation = tile_augmentation
        self.tile_paths_by_wsi = tile_paths_by_wsi
        self.num_classes = model.num_classes

    def get_tile_dataloader(self, tile_paths):
        tile_dataset = TileDataset(
            tile_paths,
            augmentation=get_augmentations_pipeline(prob=0.5),
            preprocess=self.tile_preprocess,
        )
        return DataLoader(tile_dataset,
                          batch_size=1024,
                          shuffle=False,
                          num_workers=32,
                          pin_memory=True)

    def train_epoch(self):
        """
        Executes a single training epoch and returns the average loss, supporting multi-GPU.
        """
        self.model.train()  # Set model to training mode
        running_loss = 0.0

        logging.info("Starting training epoch...")
        for batch_idx, batch in enumerate(self.dataloaders['train']):
            wsi_ids, labels = batch

            logging.info(
                f"Processing batch {batch_idx + 1}/{len(self.dataloaders['train'])}"
            )

            # Move labels to device (batch of WSI labels)
            labels = labels.to(self.device)
            batch_outputs = []

            # Loop over each WSI in the batch
            for wsi_idx, wsi_id in enumerate(wsi_ids):
                logging.info(
                    f"Processing WSI {wsi_idx + 1}/{len(wsi_ids)}: {wsi_id}")

                # Create DataLoader for the patches of the current WSI
                # we call get_augmentations_pipeline once per each WSI
                # to have the same augmentation for the whole WSI.
                tile_loader = self.get_tile_dataloader(
                    self.tile_paths_by_wsi[wsi_id])

                embeddings = self.model.embed_with_dataloader(tile_loader)

                aggregated_output, _ = self.model(embeddings)

                batch_outputs.append(
                    aggregated_output)  # Store the aggregated output

            # Stack the aggregated outputs for the entire batch
            batch_outputs = torch.stack(
                batch_outputs)  # Shape: (batch_size, num_classes)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Compute loss for the batch
            labels_one_hot = F.one_hot(labels, num_classes=self.num_classes)
            loss = self.loss_fn(batch_outputs, labels_one_hot.float())
            # Backpropagation and optimization step
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            logging.info(
                f"Completed batch {batch_idx + 1}, Loss: {loss.item():.4f}")

        return running_loss / len(self.dataloaders['train'].dataset)

    def validate_epoch(self):
        """
        Executes one epoch of validation.
        
        Returns:
            tuple: Average validation loss, predicted labels, and true labels.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        logging.info("Starting validation epoch...")
        with torch.no_grad():
            for batch_idx, (batch_patches,
                            labels) in enumerate(self.dataloaders['val']):
                logging.info(
                    f"Processing validation batch {batch_idx + 1}/{len(self.dataloaders['val'])}"
                )

                batch_patches, labels = batch_patches.to(
                    self.device), labels.to(self.device)

                # Forward pass: embeddings from patches, then predictions
                embeddings = self.model.embed_with_dataloader(batch_patches)
                outputs, _ = self.model(embeddings)

                # Compute loss
                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item() * batch_patches.size(0)

                # Collect predictions and labels
                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())

                logging.info(
                    f"Completed validation batch {batch_idx + 1}, Loss: {loss.item():.4f}"
                )

        epoch_loss = running_loss / len(self.dataloaders['val'].dataset)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        logging.info(
            f"Validation epoch completed, Average Loss: {epoch_loss:.4f}")
        return epoch_loss, all_preds, all_labels

    def train(self, num_epochs):
        """
        Trains the MIL model for a specified number of epochs.
        
        Args:
            num_epochs (int): Number of epochs to train the model.
        """
        for epoch in range(num_epochs):
            logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            # Training phase
            train_loss = self.train_epoch()

            # Validation phase
            val_loss, val_preds, val_labels = self.validate_epoch()

            logging.info(
                f"Epoch {epoch + 1} completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    def evaluate(self, dataloader: DataLoader):
        """
        Evaluates the MIL model on a test dataset.
        
        Args:
            dataloader (DataLoader): DataLoader for the test set.
        
        Returns:
            tuple: Predicted scores and, if applicable, attention weights.
        """
        self.model.eval()
        all_preds = []
        all_attentions = []

        logging.info("Starting evaluation...")
        with torch.no_grad():
            for batch_idx, batch_patches in enumerate(dataloader):
                logging.info(
                    f"Processing evaluation batch {batch_idx + 1}/{len(dataloader)}"
                )

                batch_patches = batch_patches.to(self.device)

                # Forward pass: embeddings from patches, then predictions
                embeddings = self.model.embed_with_dataloader(batch_patches)
                preds, attention = self.model(embeddings)

                all_preds.append(preds.cpu().numpy())
                if attention is not None:
                    all_attentions.append(attention.cpu().numpy())

        logging.info("Evaluation completed.")
        return np.concatenate(all_preds), (np.concatenate(all_attentions)
                                           if all_attentions else None)
