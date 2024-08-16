import torch.nn as nn

class MILLoss(nn.Module):
    def __init__(self, pos_weight=None):
        """
        A custom loss function for Multiple Instance Learning (MIL).
        
        Args:
            pos_weight (torch.Tensor, optional): A weight of positive examples for unbalanced datasets.
        """
        super(MILLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, outputs, labels):
        """
        Forward pass for the MIL loss function.

        Args:
            outputs (torch.Tensor): Model predictions.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Loss value.
        """
        return self.loss_fn(outputs, labels)
