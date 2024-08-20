import torch
from torch import nn
from torchvision import models


class BaseMILModel(nn.Module):

    def __init__(self, feature_extractor_type: str,
                 feature_extractor_params: dict,
                 aggregation_model_params: dict):
        super(BaseMILModel, self).__init__()
        self.feature_extractor = None
        self.aggregation_model = None

    def forward(self, bag_images):
        # Extract features for each instance in the bag
        bag_features = [
            self.feature_extractor(image.unsqueeze(0)) for image in bag_images
        ]  # Batch of features
        bag_features = torch.stack(bag_features, dim=0)

        # Aggregate features using the aggregation model
        output = self.aggregation_model(bag_features)
        return output
