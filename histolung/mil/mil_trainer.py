import logging

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from histolung.mil.data_loader import TileDataset, HDF5EmbeddingDataset
from histolung.models.models import MILModel

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class BaseMILTrainer:
    """
    BaseMILTrainer is a base class that provides common logic for the training loop, 
    validation, and optimizer steps. It is meant to be inherited by specific trainers 
    that handle different types of data inputs, such as raw tiles or precomputed embeddings.
    
    This class handles the shared training logic, including:
    - Backpropagation
    - Loss calculation
    - Optimizer step
    - Validation
    """

    def __init__(self,
                 model: MILModel,
                 dataloaders: dict[str, DataLoader],
                 optimizer,
                 loss_fn,
                 training_cfg: dict = None,
                 device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.dataloaders = dataloaders
        self.num_classes = model.num_classes
        self.set_scheduler(training_cfg)

    def set_scheduler(self, training_cfg):
        if not training_cfg or "scheduler" not in training_cfg:
            self.scheduler = None
            return

        if training_cfg["scheduler"]:
            scheduler_name = training_cfg["lr_scheduler"]
            scheduler_args = training_cfg["lr_scheduler_args"]

            # Dynamically get the scheduler class
            try:
                scheduler_class = getattr(lr_scheduler, scheduler_name)
            except AttributeError:
                raise ValueError(
                    f"Invalid scheduler name: {scheduler_name}. Please check the available schedulers in PyTorch."
                )

            # Instantiate the scheduler with the given arguments
            self.scheduler = scheduler_class(self.optimizer, **scheduler_args)
        else:
            self.scheduler = None

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
            for wsi_id in wsi_ids:

                # Create DataLoader for the patches of the current WSI
                # we call get_augmentations_pipeline once per each WSI
                # to have the same augmentation for the whole WSI.

                embeddings = self.get_embeddings(wsi_id)

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
            for batch_idx, batch in enumerate(self.dataloaders['val']):
                wsi_ids, labels = batch
                logging.info(
                    f"Processing validation batch {batch_idx + 1}/{len(self.dataloaders['val'])}"
                )

                labels = labels.to(self.device)

                # Forward pass: embeddings from patches, then predictions
                batch_outputs = []
                for wsi_id in wsi_ids:
                    embeddings = self.get_embeddings(wsi_id)
                    outputs, _ = self.model(embeddings)
                    batch_outputs.append(outputs)

                # Stack the aggregated outputs for the entire batch
                batch_outputs = torch.stack(batch_outputs)
                # Compute loss
                labels_one_hot = F.one_hot(labels,
                                           num_classes=self.num_classes)
                loss = self.loss_fn(batch_outputs, labels_one_hot.float())
                running_loss += loss.item() * batch_outputs.size(0)

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
        # Lists to track the losses
        training_losses = []
        validation_losses = []
        epochs_without_improvement = 0

        # Create the tqdm progress bar
        with tqdm(total=num_epochs, desc="Training Progress",
                  unit="epoch") as pbar:
            for epoch in range(num_epochs):
                # Training phase
                train_loss = self.train_epoch()
                training_losses.append(train_loss)

                # Validation phase
                val_loss, val_preds, val_labels = self.validate_epoch()
                validation_losses.append(val_loss)

                # Update tqdm bar with epoch number, train loss, and val loss
                pbar.set_postfix({
                    'Train Loss': f'{train_loss:.4f}',
                    'Val Loss': f'{val_loss:.4f}'
                })
                pbar.update(1)

                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if (self.early_stop_patience and epochs_without_improvement
                            >= self.early_stop_patience):
                        print(
                            f"Stopping early after {epoch + 1} epochs without improvement."
                        )
                        break

                # Scheduler step (after epoch)
                if self.scheduler:
                    self.scheduler.step(val_loss)

                # Update tqdm progress bar
                pbar.set_postfix({
                    'Train Loss': f'{train_loss:.4f}',
                    'Val Loss': f'{val_loss:.4f}'
                })
                pbar.update(1)

        return training_losses, validation_losses

    def get_embeddings(self, wsi_id):
        """
        To be implemented by subclasses for specific data extraction logic.
        Each subclass defines how to get the embeddings and labels.
        
        Args:
            batch_data (tuple): Batch data from DataLoader.
        
        Returns:
            tuple: Embeddings and labels.
        """
        raise NotImplementedError


class TileMILTrainer(BaseMILTrainer):
    """
    TileMILTrainer is designed to train MIL models directly from raw tile data.
    It computes embeddings for each tile on-the-fly and applies augmentation and preprocessing.
    
    **Note:** This trainer is kept for completeness but is not recommended for practical 
    use due to the long training times when computing embeddings dynamically. It's more
    efficient to use `EmbeddingMILTrainer`, which works with precomputed embeddings.
    """

    def __init__(self,
                 model,
                 dataloaders,
                 optimizer,
                 loss_fn,
                 device='cuda',
                 training_cfg=None,
                 tile_preprocess=None,
                 tile_paths_by_wsi=None,
                 tile_augmentation=None):
        super().__init__(model,
                         dataloaders,
                         optimizer,
                         loss_fn,
                         device=device,
                         training_cfg=training_cfg)
        self.dataloaders = dataloaders
        self.tile_preprocess = tile_preprocess
        self.tile_augmentation = tile_augmentation
        if tile_paths_by_wsi is None:
            raise RuntimeError(
                "You must provide a mapping to the path of the patches "
                "in the tile_paths_by_wsi kwargs")
        else:
            self.tile_paths_by_wsi = tile_paths_by_wsi

    def get_embeddings(self, wsi_id):
        """
        Extracts the embeddings for a batch by computing them dynamically from raw tile data.
        
        Args:
            batch_data (tuple): Tuple containing (wsi_id, tile_paths, label).
        
        Returns:
            tuple: Embeddings for the WSI and its label.
        """
        tile_paths = self.tile_paths_by_wsi[wsi_id]
        tile_loader = self.create_tile_loader(tile_paths)
        embeddings = self.model.embed_with_dataloader(tile_loader)
        return embeddings

    def create_tile_loader(self, tile_paths):
        """
        Create a DataLoader for the tiles of a given WSI.
        
        Args:
            tile_paths (list): List of file paths for WSI tiles.
        
        Returns:
            DataLoader: DataLoader for the WSI tiles.
        """
        tile_dataset = TileDataset(
            tile_paths,
            augmentation=self.tile_augmentation,
            preprocess=self.tile_preprocess,
        )
        return DataLoader(tile_dataset,
                          batch_size=32,
                          shuffle=False,
                          num_workers=4)


class EmbeddingMILTrainer(BaseMILTrainer):
    """
    EmbeddingMILTrainer is optimized for training MIL models using precomputed embeddings. 
    It skips tile augmentation and preprocessing, and instead loads embeddings from an HDF5 
    file or other storage. This approach is significantly faster and recommended for practical 
    training purposes.
    """

    def __init__(self,
                 model,
                 dataloaders,
                 optimizer,
                 loss_fn,
                 device='cuda',
                 training_cfg: dict = None,
                 hdf5_file=None):
        super().__init__(
            model,
            dataloaders,
            optimizer,
            loss_fn,
            device,
            training_cfg=training_cfg,
        )
        self.dataloaders = dataloaders
        self.hdf5_file = hdf5_file

    def get_embeddings(self, wsi_id):
        """
        Loads the precomputed embeddings for a batch from the HDF5 file.
        
        Args:
            batch_data (tuple): Tuple containing (wsi_id, label).
        
        Returns:
            tuple: Embeddings for the WSI and its label.
        """
        embeddings = torch.tensor(self.hdf5_file['embeddings'][wsi_id][:]).to(
            self.device)
        return embeddings
