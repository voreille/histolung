import torch
import torch.nn as nn
from torchvision import models
from .base_model import BaseModel


class UnifiedMILModel(BaseModel):

    def __init__(self, feature_extractor_type: str,
                 feature_extractor_params: dict,
                 aggregation_model_params: dict):
        super(UnifiedMILModel, self).__init__()
        self.feature_extractor = self._build_feature_extractor(
            feature_extractor_type, feature_extractor_params)
        self.aggregation_model = self._build_aggregation_model(
            aggregation_model_params)

    def _build_feature_extractor(self, model_type: str, model_params: dict):
        if model_type == 'resnet':
            model = models.resnet50(
                pretrained=model_params.get('pretrained', True))
            # Remove the last classification layer to use it as a feature extractor
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model

    def _build_aggregation_model(self, aggregation_params: dict):
        input_dim = aggregation_params['input_dim']
        output_dim = aggregation_params['output_dim']
        return nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(),
                             nn.Linear(256, output_dim))

    def forward(self, bag_images):
        # Extract features for each instance in the bag
        bag_features = [
            self.feature_extractor(image.unsqueeze(0)) for image in bag_images
        ]  # Batch of features
        bag_features = torch.stack(bag_features, dim=0)

        # Aggregate features using the aggregation model
        output = self.aggregation_model(bag_features)
        return output

    def get_layer(self, layer_name):
        """
        Returns a specific layer of the feature extractor for explainability methods.
        """
        if layer_name in dict(self.feature_extractor.named_children()):
            return dict(self.feature_extractor.named_children())[layer_name]
        else:
            raise ValueError(
                f"Layer {layer_name} not found in the feature extractor.")
