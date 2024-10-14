from collections.abc import Iterable

from torch import nn
import torchvision.models as models
import torch
import torch.nn.functional as F


class PretrainedModelLoader:

    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 freeze=False,
                 num_freezed_layers=0,
                 dropout=0.0,
                 embedding_bool=False,
                 pool_algorithm=None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_freezed_layers = num_freezed_layers
        self.dropout = dropout
        self.embedding_bool = embedding_bool
        self.pool_algorithm = pool_algorithm

        # Load the appropriate pre-trained model and its input feature size
        (self.net, self.input_features,
         self.resize_param) = self.load_pretrained_model()

    def load_pretrained_model(self):
        """Loads the pre-trained model based on the model name and returns it."""
        model_dict = {
            "resnet50":
            (models.resnet50, models.ResNet50_Weights.DEFAULT, 224),
            "resnet34":
            (models.resnet34, models.ResNet34_Weights.DEFAULT, 224),
            "resnet101":
            (models.resnet101, models.ResNet101_Weights.DEFAULT, 224),
            "convnext": (models.convnext_small, 'DEFAULT', 224),
            "swin": (models.swin_v2_t, models.Swin_V2_T_Weights.DEFAULT, 224),
            "efficient": (models.efficientnet_b0, 'DEFAULT', 224),
        }

        if self.model_name.lower() not in model_dict:
            raise ValueError(
                f"Invalid model name: {self.model_name}. Choose from {list(model_dict.keys())}"
            )

        model_class, weights, resize_param = model_dict[
            self.model_name.lower()]
        model = model_class(weights=weights)

        # Extract input features from the appropriate layer
        input_features = self.get_input_features(model)

        return model, input_features, resize_param

    def get_input_features(self, model):
        """Extracts the input features based on the model's architecture."""
        if hasattr(model, 'fc'):
            return model.fc.in_features
        elif hasattr(model, 'classifier'):
            return model.classifier[-1].in_features
        elif hasattr(model, 'head'):
            return model.head.in_features
        else:
            raise AttributeError(
                "The model does not have a recognized fully connected or classifier layer."
            )

    @staticmethod
    def set_parameter_requires_grad(model, num_freezed_layers):
        """Freezes the layers of the model based on the number of layers to freeze."""
        for k, child in enumerate(model.children()):
            if k == num_freezed_layers:
                break
            for param in child.parameters():
                param.requires_grad = False


class MILModel(nn.Module):

    def __init__(self, model_loader: PretrainedModelLoader,
                 hidden_space_len: int, cfg):
        super(MILModel, self).__init__()
        self.model_loader = model_loader
        self.hidden_space_len = hidden_space_len
        self.cfg = cfg

        # TODO: change this self.net and self.conv_layers
        #       does it even make sense to have the two?
        self.net = self.model_loader.net

        # Get the pre-trained conv layers
        self.conv_layers = nn.Sequential(
            *list(self.model_loader.net.children())[:-1])

        # Parallelize on multiple GPUs if available
        if torch.cuda.device_count() > 1:
            self.conv_layers = nn.DataParallel(self.conv_layers,
                                               device_ids=[0])

        self.embedding_bool = self.model_loader.embedding_bool
        self.fc_input_features = self.model_loader.input_features
        self.num_classes = self.model_loader.num_classes
        self.pool_algorithm = self.model_loader.pool_algorithm

        self.E = self.hidden_space_len
        self.K = self.num_classes
        self.D = self.hidden_space_len
        if self.embedding_bool:
            self.setup_embedding_layers()
        else:
            self.fc = nn.Linear(in_features=self.fc_input_features,
                                out_features=self.num_classes)

        self.embedding_fc = torch.nn.Linear(self.E, self.K)

        if self.pool_algorithm == "attention":
            self.setup_attention_pooling()

        self.dropout = nn.Dropout(p=self.model_loader.dropout)

        # self.relu = torch.nn.ReLU()
        # self.activation = self.relu
        self.activation = torch.nn.ReLU()
        self.LayerNorm = torch.nn.LayerNorm(self.E * self.K, eps=1e-5)

    def setup_embedding_layers(self):
        """Sets up embedding layers based on the selected model."""

        self.embedding = nn.Linear(in_features=self.fc_input_features,
                                   out_features=self.E)

        # REMOVE: it's there from old code
        self.post_embedding = nn.Linear(in_features=self.E,
                                        out_features=self.E)

    def setup_attention_pooling(self):
        """Sets up attention pooling layers."""
        self.attention = nn.Sequential(nn.Linear(self.E, self.D), nn.Tanh(),
                                       nn.Linear(self.D, self.K))
        if "AChannel" in self.cfg.data_augmentation.featuresdir:
            self.attention_channel = nn.Sequential(nn.Linear(self.E, self.D),
                                                   nn.Tanh(),
                                                   nn.Linear(self.D, 1))

        self.embedding_before_fc = nn.Linear(self.E * self.K, self.E)

    def forward(self, x):
        """
        Forward pass for the MIL model using precomputed embeddings.

        Args:
            x (torch.Tensor): The input tensor containing the embeddings of the patches 
                            for a WSI, with shape (num_patches, embedding_dim). If 
                            `self.embedding_bool` is True, `x` will be passed through 
                            additional embedding layers; otherwise, it will be used 
                            directly in the attention mechanism.

        Returns:
            Y_prob (torch.Tensor): The output probabilities for the WSI, representing the 
                                predicted class distribution, with shape (num_classes,).
            A (torch.Tensor): The attention weights for each patch, used for aggregating 
                            the patch-level embeddings into a WSI-level representation, 
                            with shape (num_patches,).
        
        Behavior:
        - If `self.embedding_bool` is True, the input `x` will first be passed through 
        an additional embedding layer (`self.embedding`) before further processing.
        - The model then applies attention pooling over the embeddings or post-embedding 
        features using the learned attention mechanism (`self.attention`).
        - The attention weights are computed and used to aggregate the patch embeddings 
        into a single WSI-level embedding (`wsi_embedding`).
        - If the "AChannel" option is enabled in the data augmentation configuration, 
        an additional channel-specific attention mechanism is applied to the WSI-level 
        embedding.
        - The resulting WSI embedding is passed through a fully connected layer and 
        activation function to produce the final prediction probabilities.
        """
        x = x.view(-1, self.fc_input_features)

        if self.embedding_bool:
            embedding_layer = self.embedding(x)
            features_to_return = embedding_layer

            # REMOVE: it's there from old code
            embedding_layer = self.dropout(embedding_layer)
        else:
            features_to_return = x

        # Attention pooling
        A = self.attention(features_to_return)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        wsi_embedding = torch.mm(A, features_to_return)

        if "AChannel" in self.cfg.data_augmentation.featuresdir:
            attention_channel = self.attention_channel(wsi_embedding)
            attention_channel = torch.transpose(attention_channel, 1, 0)
            attention_channel = F.softmax(attention_channel, dim=1)
            cls_img = torch.mm(attention_channel, wsi_embedding)
        else:
            cls_img = wsi_embedding.view(-1, self.E * self.K)
            cls_img = self.embedding_before_fc(cls_img)

        cls_img = self.activation(cls_img)
        cls_img = self.dropout(cls_img)

        Y_prob = self.embedding_fc(cls_img)
        Y_prob = torch.squeeze(Y_prob)

        return Y_prob, A

    def embed_with_dataloader(self, dataloader):
        """
        Forward pass for a WSI using a DataLoader.
        This method processes the WSI in batches using a DataLoader and aggregates the embeddings.
        """
        all_embeddings = []
        self.eval()  # Switch to evaluation mode

        with torch.no_grad():
            for batch_patches in dataloader:
                batch_patches = batch_patches.to(
                    next(self.parameters()).device)
                embeddings = self.conv_layers(batch_patches)
                embeddings = embeddings.view(-1, self.hidden_space_len)
                all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings

    def process_dataloader(self, dataloader):
        pred_scores, attentions = self.forward(
            self.embed_with_dataloader(dataloader))
        pred_scores = pred_scores.detach().cpu().numpy()
        attentions = attentions.detach().cpu().numpy()
        return pred_scores, attentions
