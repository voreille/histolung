import os
from pathlib import Path
from abc import ABC, abstractmethod

import torch.nn as nn
import torch
import torchvision.transforms as T
from torchvision import models
import timm
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv


class BaseFeatureExtractor(ABC, nn.Module):

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        self.model = None
        self._frozen = False
        self.feature_dim = None

    def forward(self, x):
        return self.model(x)

    @abstractmethod
    def load_model(self):
        pass

    def get_preprocessing(self, data_cfg):
        image_size = data_cfg["image_size"]
        mean = data_cfg["mean"]
        std = data_cfg["std"]
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    @staticmethod
    def is_resnet(model_name: str):
        """
        Check if the model is a ResNet variant.
        
        Args:
            model_name (str): Name of the model.
        
        Returns:
            bool: True if it's a ResNet model, False otherwise.
        """
        return model_name in {"resnet50", "resnet101", "resnet34"}

    @staticmethod
    def get_feature_extractor(model_name: str, **kwargs):
        """
        Returns the appropriate FeatureExtractor class based on the name provided.
        
        Args:
            name (str): Name of the feature extractor model.
        
        Returns:
            class: The FeatureExtractor class.
        
        Raises:
            ValueError: If the feature extractor name is not recognized.
        """
        if BaseFeatureExtractor.is_resnet(model_name):
            return ResNetFeatureExtractor(model_name, **kwargs)

        extractor_classes = {
            # "convnext": ConvNextFeatureExtractor,
            "UNI": UNIFeatureExtractor,
        }

        if model_name not in extractor_classes:
            raise ValueError(
                f"Invalid feature extractor name: {model_name}. Choose from {list(extractor_classes.keys())}"
            )

        return extractor_classes[model_name](model_name, **kwargs)

    @property
    def frozen(self):
        print("getter method called")
        return self._frozen

    # a setter function
    @frozen.setter
    def frozen(self, value: bool):
        """
        Setter for the freeze_weights state. If set to True, freezes the weights; if False, unfreezes.
        """
        if value:
            self.freeze_weights()
        else:
            self.unfreeze_weights()

    def freeze_weights(self):
        """
        Freezes all layers in the feature extractor by setting requires_grad to False.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        self._frozen = True  # Update state

    def unfreeze_weights(self):
        """
        Unfreezes all layers in the feature extractor by setting requires_grad to True.
        """
        for param in self.model.parameters():
            param.requires_grad = True
        self._frozen = False  # Update state


class UNIFeatureExtractor(nn.Module):

    def __init__(self,
                 model_name: str,
                 weights_filepath=None,
                 freeze_weights=True):
        super(UNIFeatureExtractor, self).__init__()
        if weights_filepath is None:
            raise RuntimeError(
                "You need to provide a filepath for the weights of UNI model")
        self.weights_filepath = Path(weights_filepath)
        self.model_name = model_name
        self.model = None
        self.feature_dim = 1024
        # Ensure the directory exists
        self.weights_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Load environment variables from .env
        load_dotenv()

        # Load the model weights
        self.load_model()
        self.frozen = freeze_weights

    def load_model(self):
        """Load the model, downloading the weights if not present."""
        # Check if the weights file already exists
        if not self.weights_filepath.exists():
            print(f"Downloading model weights to {self.weights_filepath}...")
            token = os.getenv("HUGGING_FACE_TOKEN")

            if not token:
                raise ValueError(
                    "HUGGING_FACE_TOKEN not found in environment variables")

            hf_hub_download(
                repo_id="MahmoodLab/UNI",
                filename=self.weights_filepath.name,
                local_dir=self.weights_filepath.parent,
                force_download=True,
                token=token,
            )
            print(f"Downloaded model weights to {self.weights_filepath}.")
        else:
            print(f"Loading model weights from {self.weights_filepath}...")

        # Initialize the model using timm
        self.model = timm.create_model("vit_large_patch16_224",
                                       img_size=224,
                                       patch_size=16,
                                       init_values=1e-5,
                                       num_classes=0,
                                       dynamic_img_size=True)

        # Load the weights
        self.model.load_state_dict(torch.load(self.weights_filepath,
                                              map_location="cpu"),
                                   strict=True)
        self.model.eval()  # Set model to evaluation mode

    def forward(self, x):
        return self.model(x)


class ResNetFeatureExtractor(BaseFeatureExtractor):

    def __init__(
        self,
        model_name: str,
        freeze_weights=True,
    ):
        super(ResNetFeatureExtractor, self).__init__()
        self.model_name = model_name
        self.model = None
        self.feature_dim = None
        self.load_model()
        self.frozen = freeze_weights

    def load_model(self):
        model_dict = {
            "resnet50":
            (models.resnet50, models.ResNet50_Weights.DEFAULT, 224),
            "resnet34":
            (models.resnet34, models.ResNet34_Weights.DEFAULT, 224),
            "resnet101":
            (models.resnet101, models.ResNet101_Weights.DEFAULT, 224),
        }
        model_class, weights, self.image_size = model_dict[self.model_name]
        self.model = model_class(weights=weights)

        self.feature_dim = self.model.fc.in_features if hasattr(
            self.model, 'fc') else self.model.classifier[-1].in_features

        if hasattr(self.model, 'fc'):
            self.model.fc = nn.Identity()
        elif hasattr(self.model, 'classifier'):
            self.model.classifier = nn.Identity()

        if self.freeze_weights:
            self.freeze_weights()

        # if self.fine_tune_last_n_layers > 0:
        #     self.unfreeze_last_n_layers(self.fine_tune_last_n_layers)

    def unfreeze_last_n_layers(self, n: int):
        total_layers = list(self.model.parameters())
        for param in total_layers[-n:]:
            param.requires_grad = True
