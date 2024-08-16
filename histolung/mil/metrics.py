import torch


def accuracy(outputs, labels):
    """
    Computes the accuracy of the model's predictions.

    Args:
        outputs (torch.Tensor): Model predictions (logits).
        labels (torch.Tensor): Ground truth labels.

    Returns:
        float: Accuracy score.
    """
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def precision(outputs, labels):
    """
    Computes the precision of the model's predictions.

    Args:
        outputs (torch.Tensor): Model predictions (logits).
        labels (torch.Tensor): Ground truth labels.

    Returns:
        float: Precision score.
    """
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    true_positives = (predicted * labels).sum().item()
    predicted_positives = predicted.sum().item()

    if predicted_positives == 0:
        return 0.0

    return true_positives / predicted_positives


def recall(outputs, labels):
    """
    Computes the recall of the model's predictions.

    Args:
        outputs (torch.Tensor): Model predictions (logits).
        labels (torch.Tensor): Ground truth labels.

    Returns:
        float: Recall score.
    """
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    true_positives = (predicted * labels).sum().item()
    actual_positives = labels.sum().item()

    if actual_positives == 0:
        return 0.0

    return true_positives / actual_positives
