from torch import nn
import torch
import torch.nn.functional as F


# Example of attention-based aggregator
class AttentionAggregator(nn.Module):

    def __init__(self, input_dim):
        super(AttentionAggregator, self).__init__()
        self.attention = nn.Sequential(nn.Linear(input_dim, input_dim),
                                       nn.Tanh(), nn.Linear(input_dim, 1))

    def forward(self, embeddings):
        A = self.attention(embeddings)
        A = torch.softmax(A, dim=0)
        aggregated_embedding = torch.sum(A * embeddings, dim=0)
        return aggregated_embedding, A


# Example of mean pooling aggregator
class MeanPoolingAggregator(nn.Module):

    def __init__(self):
        super(MeanPoolingAggregator, self).__init__()

    def forward(self, embeddings):
        return torch.mean(embeddings,
                          dim=0), None  # No attention weights to return


class MILModel(nn.Module):

    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 aggregator: nn.Module,
                 embedding_bool=False,
                 dropout=0.0):
        super(MILModel, self).__init__()
        self.embedding_bool = embedding_bool
        self.num_classes = num_classes

        # Load pre-trained model
        self.conv_layers, self.fc_input_features, _ = load_pretrained_model(
            model_name)

        # Remove the final fully connected layers from the pre-trained model (keeping conv layers)
        self.conv_layers = nn.Sequential(
            *list(self.conv_layers.children())[:-1])

        # Define the embedding layer if needed
        if self.embedding_bool:
            self.embedding = nn.Linear(self.fc_input_features,
                                       self.fc_input_features)

        # Set the aggregator (attention-based, mean pooling, or others)
        self.aggregator = aggregator

        # Define the final classifier
        self.embedding_fc = nn.Linear(self.fc_input_features, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def extract_embeddings(self, x):
        """
        Extracts the embeddings from input patches.
        
        Args:
            x (torch.Tensor): Input tensor of patches.
            
        Returns:
            torch.Tensor: The extracted embeddings.
        """
        x = x.view(-1, self.fc_input_features)
        if self.embedding_bool:
            x = self.embedding(x)
        return x

    def forward(self, x):
        """
        Forward pass for MIL model.
        Embedding extraction and aggregation are separated for memory efficiency.
        
        Args:
            x (torch.Tensor): Input embeddings.
        
        Returns:
            torch.Tensor: Prediction probabilities.
        """
        embeddings = self.extract_embeddings(x)
        wsi_embedding, attention_weights = self.aggregator(embeddings)

        # Final classification
        wsi_embedding = self.activation(wsi_embedding)
        wsi_embedding = self.dropout(wsi_embedding)
        Y_prob = self.embedding_fc(wsi_embedding)

        return Y_prob, attention_weights

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
                embeddings = self.conv_layers(batch_patches)
                embeddings = embeddings.view(-1, self.fc_input_features)
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
        pred_scores, _ = self.forward(self.embed_with_dataloader(dataloader))
        return pred_scores.detach().cpu().numpy()
