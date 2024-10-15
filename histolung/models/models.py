from typing import Union
from pathlib import Path
import yaml

from torch import nn
import torch
import torchvision.models as models

from .feature_extractor import BaseFeatureExtractor


def load_pretrained_model(model_name: str, keep_last_layer=False):
    """Loads the specified pre-trained model and returns the model and input features."""
    model_dict = {
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 224),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 224),
        "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, 224),
        "convnext": (models.convnext_small, 'DEFAULT', 224),
        "swin": (models.swin_v2_t, models.Swin_V2_T_Weights.DEFAULT, 224),
        "efficient": (models.efficientnet_b0, 'DEFAULT', 224),
    }

    if model_name.lower() not in model_dict:
        raise ValueError(
            f"Invalid model name: {model_name}. Choose from {list(model_dict.keys())}"
        )

    model_class, weights, image_size = model_dict[model_name.lower()]
    model = model_class(weights=weights)

    # Extract input features from the appropriate layer before modifying the last layer
    feature_dim = model.fc.in_features if hasattr(
        model, 'fc') else model.classifier[-1].in_features

    # Remove the last layer if keep_last_layer is False
    if not keep_last_layer:
        if hasattr(model, 'fc'):  # For models like ResNet
            model.fc = nn.Identity()  # Replaces the fully connected layer
        elif hasattr(model,
                     'classifier'):  # For models like EfficientNet, ConvNext
            model.classifier = nn.Identity()

    return model, feature_dim, image_size


# class FeatureExtractor(nn.Module):

#     def __init__(self,
#                  model_name: str,
#                  keep_last_layer=False,
#                  freeze_weights=True,
#                  fine_tune_last_n_layers=0):
#         super(FeatureExtractor, self).__init__()
#         self.model, self.feature_dim, self.image_size = load_pretrained_model(
#             model_name=model_name, keep_last_layer=keep_last_layer)

#         # Freeze all layers by default if freeze_weights is set to True
#         if freeze_weights:
#             self.freeze_all_layers()

#         # Fine-tune the last `n` layers if specified
#         if fine_tune_last_n_layers > 0:
#             self.unfreeze_last_n_layers(fine_tune_last_n_layers)

#     def freeze_all_layers(self):
#         """
#         Freezes all layers in the feature extractor by setting requires_grad to False.
#         """
#         for param in self.model.parameters():
#             param.requires_grad = False

#     def unfreeze_last_n_layers(self, n: int):
#         """
#         Unfreezes the last `n` layers of the feature extractor, based on its architecture.

#         Args:
#             n (int): Number of layers from the end to unfreeze.
#         """
#         # This is an architecture-specific fine-tuning method.
#         # Unfreeze the last `n` layers based on the architecture.
#         total_layers = list(self.model.parameters())
#         for param in total_layers[-n:]:
#             param.requires_grad = True

#         # Additionally, handle specific types of layers like BatchNorm
#         for layer in self.model.modules():
#             if isinstance(layer, nn.BatchNorm2d) and not any(
#                     p.requires_grad for p in layer.parameters()):
#                 layer.eval()  # Keep BatchNorm in eval mode for frozen layers

#     def forward(self, x):
#         return self.model(x)


# Example of attention-based aggregator
class BaseAggregator(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes=2, dropout=0.2):
        super(BaseAggregator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout


class AttentionAggregator(BaseAggregator):

    def __init__(self, input_dim, hidden_dim, num_classes=2, dropout=0.2):
        super(AttentionAggregator, self).__init__(
            input_dim,
            hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
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
        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=num_classes,
        )

    def forward(self, x):
        x = self.projection_layer(x)
        attention = self.attention(x)
        attention = torch.transpose(attention, 1, 0)
        aggregated_embedding = torch.mm(attention, x)
        aggregated_embedding = aggregated_embedding.view(
            -1,
            self.hidden_dim * self.num_classes,
        )
        output = self.pre_fc_layer(aggregated_embedding)
        output = self.fc(output)
        output = torch.squeeze(output)

        return output, attention


# Example of mean pooling aggregator
class MeanPoolingAggregator(nn.Module):

    def __init__(self):
        super(MeanPoolingAggregator, self).__init__()

    def forward(self, embeddings):
        return torch.mean(embeddings,
                          dim=0), None  # No attention weights to return


class MILModel(nn.Module):

    def __init__(
        self,
        feature_extractor: BaseFeatureExtractor,
        aggregator: BaseAggregator,
    ):
        super(MILModel, self).__init__()
        self.num_classes = aggregator.num_classes
        self.feature_dim = feature_extractor.feature_dim

        self.feature_extractor = feature_extractor
        self.aggregator = aggregator

    @staticmethod
    def from_config(cfg: Union[dict, Path, str]):
        """
        Builds a MILModel instance from a configuration dictionary or YAML file.
        
        Args:
            cfg (Union[dict, Path, str]): Configuration dictionary or path to a YAML file.
        
        Returns:
            MILModel: Configured MILModel instance.
        """
        if isinstance(cfg, Path) or isinstance(cfg, str):
            with open(cfg, 'r') as file:
                cfg = yaml.safe_load(file)

        # Load parameters from config
        model_cfg = cfg['model']

        # Model-specific parameters
        feature_extractor_name = model_cfg['feature_extractor']
        aggregator_type = model_cfg['aggregator']
        projection_dim = model_cfg['projection_dim']
        num_classes = model_cfg['num_classes']

        # Instantiate the feature extractor
        feature_extractor = BaseFeatureExtractor.get_feature_extractor(
            feature_extractor_name, **model_cfg["feature_extractor_kwargs"])

        # Instantiate the aggregator
        if aggregator_type == 'attention':
            aggregator = AttentionAggregator(
                input_dim=feature_extractor.feature_dim,
                hidden_dim=projection_dim,
                num_classes=num_classes)
        else:
            aggregator = MeanPoolingAggregator()

        # Build the MILModel instance
        model = MILModel(feature_extractor=feature_extractor,
                         aggregator=aggregator)

        return model

    def forward(self, x):
        """
        Forward pass for MIL model.
        Embedding extraction and aggregation are separated for memory efficiency.
        
        Args:
            x (torch.Tensor): Input embeddings.
        
        Returns:
            torch.Tensor: Prediction probabilities.
        """
        return self.aggregator(x)

    def embed_with_dataloader(self, dataloader):
        """
        Processes a WSI using a DataLoader, extracting embeddings in batches.
        
        Args:
            dataloader (DataLoader): DataLoader for batch-wise processing of patches.
        
        Returns:
            torch.Tensor: Extracted embeddings.
        """
        all_embeddings = []
        self.eval()  # Switch to evaluation mode

        with torch.no_grad():
            for batch_patches in dataloader:
                batch_patches = batch_patches.to(
                    next(self.parameters()).device)
                embeddings = self.feature_extractor(batch_patches)
                all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def process_dataloader(self, dataloader):
        """
        Processes the WSI by first embedding the patches and then aggregating.
        
        Args:
            dataloader (DataLoader): DataLoader for batch-wise processing.
        
        Returns:
            numpy.ndarray: Prediction scores.
        """
        pred_scores, attention = self.forward(
            self.embed_with_dataloader(dataloader))
        return (
            pred_scores.detach().cpu().numpy(),
            attention.detach().cpu().numpy(),
        )
