import torch

class MILTrainer:
    def __init__(self, model, criterion, optimizer, device="cpu"):
        """
        MIL Trainer class that handles training and evaluation for the MIL model.

        Args:
            model (torch.nn.Module): The unified MIL model (feature extractor + aggregation).
            criterion (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): Optimizer for training the model.
            device (str): Device to run training on ('cpu' or 'cuda').
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        # Move model to the appropriate device
        self.model.to(self.device)

    def train_step(self, bag_images, labels):
        """
        Executes a single training step.

        Args:
            bag_images (list of torch.Tensor): List of images forming the bag (instances).
            labels (torch.Tensor): Labels for the bags (e.g., binary labels).

        Returns:
            float: The training loss for this step.
        """
        self.model.train()

        # Move data to the appropriate device
        bag_images = [image.to(self.device) for image in bag_images]
        labels = labels.to(self.device)

        # Forward pass
        outputs = self.model(bag_images)

        # Compute loss
        loss = self.criterion(outputs, labels)

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, val_loader):
        """
        Validates the model on the validation set.

        Args:
            val_loader (torch.utils.data.DataLoader): Validation data loader.

        Returns:
            dict: Dictionary containing validation loss and metrics.
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for bag_images, labels in val_loader:
                # Move data to the appropriate device
                bag_images = [image.to(self.device) for image in bag_images]
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(bag_images)
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # Compute accuracy or other metrics
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        accuracy = correct / total

        return {"val_loss": val_loss, "accuracy": accuracy}
