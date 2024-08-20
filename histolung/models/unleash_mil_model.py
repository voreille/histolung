import torch
from torchvision import models
import torch.nn.functional as F

from histolung.models.base_mil_model import BaseMILModel  # Your base class


class UnleashMILModel(BaseMILModel):
    """
    UnleashMILModel integrates the methods from 'Unleashing the potential of digital pathology
    data by training computer-aided diagnosis models without human annotations'.
    
    This model adapts the MIL architecture for self-supervised learning on pathology data.
    """

    def __init__(self, model_name, hidden_space_len, cfg):
        super(BaseMILModel, self).__init__(model_name, hidden_space_len, cfg)

        # Incorporate the original model's logic into the new unified structure
        model = ModelOption(model_name,
                            cfg.model.num_classes,
                            freeze=cfg.model.freeze_weights,
                            num_freezed_layers=cfg.model.num_frozen_layers,
                            dropout=cfg.model.dropout,
                            embedding_bool=cfg.model.embedding_bool,
                            pool_algorithm=cfg.model.pool_algorithm)

        # Initialize the original MIL model within AutoPathMILModel
        self.whole_model = MIL_model(model, hidden_space_len, cfg)
        self.feature_extractor = self.whole_model.conv_layers
        self.aggregation_model = self.whole_model.attention

    def forward(self, x, conv_layers_out=None):
        # Forward pass using the adapted MIL model
        return s


class MIL_model(torch.nn.Module):

    def __init__(self, model, hidden_space_len, cfg):

        super(MIL_model, self).__init__()

        self.model = model
        self.fc_input_features = self.model.input_features
        self.num_classes = self.model.num_classes
        self.hidden_space_len = hidden_space_len
        self.net = self.model.net
        self.cfg = cfg

        self.conv_layers = torch.nn.Sequential(*list(self.net.children())[:-1])

        if (torch.cuda.device_count() > 1):
            # 0 para GPU buena
            self.conv_layers = torch.nn.DataParallel(self.conv_layers,
                                                     device_ids=[0])

        if self.model.embedding_bool:
            if ('resnet34' in self.model.model_name):
                self.E = self.hidden_space_len
                self.L = self.E
                self.D = self.hidden_space_len
                self.K = self.num_classes

            elif ('resnet101' in self.model.model_name):
                self.E = self.hidden_space_len
                self.L = self.E
                self.D = self.hidden_space_len
                self.K = self.num_classes

            elif ('convnext' in self.model.model_name):
                self.E = self.hidden_space_len
                self.L = self.E
                self.D = self.hidden_space_len
                self.K = self.num_classes

            self.embedding = torch.nn.Linear(
                in_features=self.fc_input_features, out_features=self.E)
            self.post_embedding = torch.nn.Linear(in_features=self.E,
                                                  out_features=self.E)

        else:
            self.fc = torch.nn.Linear(in_features=self.fc_input_features,
                                      out_features=self.num_classes)

            if ('resnet34' in self.model.model_name):
                self.L = self.fc_input_features
                self.D = self.hidden_space_len
                self.K = self.num_classes

            elif ('resnet101' in self.model.model_name):
                self.L = self.E
                self.D = self.hidden_space_len
                self.K = self.num_classes

            elif ('convnext' in self.model.model_name):
                self.L = self.E
                self.D = self.hidden_space_len
                self.K = self.num_classes

        if (self.model.pool_algorithm == "attention"):
            self.attention = torch.nn.Sequential(
                torch.nn.Linear(self.L, self.D), torch.nn.Tanh(),
                torch.nn.Linear(self.D, self.K))

            if "NoChannel" in self.cfg.data_augmentation.featuresdir:
                # print("== Attention No Channel ==")
                self.embedding_before_fc = torch.nn.Linear(
                    self.E * self.K, self.E)

            elif "AChannel" in self.cfg.data_augmentation.featuresdir:
                # print("== Attention with A Channel for multilabel ==")
                self.attention_channel = torch.nn.Sequential(
                    torch.nn.Linear(self.L, self.D), torch.nn.Tanh(),
                    torch.nn.Linear(self.D, 1))
                self.embedding_before_fc = torch.nn.Linear(self.E, self.E)

        self.embedding_fc = torch.nn.Linear(self.E, self.K)

        self.dropout = torch.nn.Dropout(p=self.model.dropout)
        # self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

        self.LayerNorm = torch.nn.LayerNorm(self.E * self.K, eps=1e-5)
        # self.activation = self.tanh
        self.activation = self.relu

    def forward(self, x, conv_layers_out):

        #if used attention pooling
        A = None
        #m = torch.nn.Softmax(dim=1)

        if x is not None:
            #print(x.shape)
            conv_layers_out = self.conv_layers(x)
            #print(x.shape)

            conv_layers_out = conv_layers_out.view(-1, self.fc_input_features)

        if self.model.embedding_bool:
            embedding_layer = self.embedding(conv_layers_out)

            #embedding_layer = self.LayerNorm(embedding_layer)
            features_to_return = embedding_layer
            embedding_layer = self.dropout(embedding_layer)

        else:
            embedding_layer = conv_layers_out
            features_to_return = embedding_layer

        A = self.attention(features_to_return)

        A = torch.transpose(A, 1, 0)

        A = F.softmax(A, dim=1)

        wsi_embedding = torch.mm(A, features_to_return)

        if "NoChannel" in self.cfg.data_augmentation.featuresdir:
            # print("== Attention No Channel ==")
            wsi_embedding = wsi_embedding.view(-1, self.E * self.K)

            cls_img = self.embedding_before_fc(wsi_embedding)

        elif "AChannel" in self.cfg.data_augmentation.featuresdir:
            # print("== Attention with A Channel for multilabel ==")
            attention_channel = self.attention_channel(wsi_embedding)

            attention_channel = torch.transpose(attention_channel, 1, 0)

            attention_channel = F.softmax(attention_channel, dim=1)

            cls_img = torch.mm(attention_channel, wsi_embedding)

            # cls_img = self.embedding_before_fc(cls_img)

        cls_img = self.activation(cls_img)

        cls_img = self.dropout(cls_img)

        Y_prob = self.embedding_fc(cls_img)

        Y_prob = torch.squeeze(Y_prob)

        return Y_prob, A


class ModelOption():

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

        if self.model_name.lower() == "resnet50":
            """ ResNet50 """
            self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            self.input_features = self.net.fc.in_features  # 2048

            # self.net.fc = nn.Sequential(nn.Dropout(p=self.dropout),
            #                             nn.Linear(input_features,
            #                                       input_features // 4),
            #                             nn.ReLU(inplace=True),
            #                             nn.Dropout(p=self.dropout),
            #                             nn.Linear(input_features // 4,
            #                                      input_features // 8),
            #                             nn.ReLU(inplace=True),
            #                             nn.Dropout(p=self.dropout),
            #                             nn.Linear(input_features // 8,
            #                                      self.num_classes))

            self.resize_param = 224

        elif self.model_name.lower() == "resnet34":
            """ ResNet34 """
            self.net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

            self.input_features = self.net.fc.in_features  # 2048

            # self.net.fc = nn.Sequential(nn.Dropout(p=self.dropout),
            #                             nn.Linear(input_features,
            #                                       input_features // 4),
            #                             nn.ReLU(inplace=True),
            #                             nn.Dropout(p=self.dropout),
            #                             nn.Linear(input_features // 4,
            #                                      input_features // 8),
            #                             nn.ReLU(inplace=True),
            #                             nn.Dropout(p=self.dropout),
            #                             nn.Linear(input_features // 8,
            #                                      self.num_classes))

            self.resize_param = 224

        elif self.model_name.lower() == "resnet101":
            """ ResNet101 """
            self.net = models.resnet101(
                weights=models.ResNet101_Weights.DEFAULT)

            self.input_features = self.net.fc.in_features  # 2048

            # self.net.fc = nn.Sequential(nn.Dropout(p=self.dropout),
            #                             nn.Linear(input_features,
            #                                       input_features // 4),
            #                             nn.ReLU(inplace=True),
            #                             nn.Dropout(p=self.dropout),
            #                             nn.Linear(input_features // 4,
            #                                      input_features // 8),
            #                             nn.ReLU(inplace=True),
            #                             nn.Dropout(p=self.dropout),
            #                             nn.Linear(input_features // 8,
            #                                      self.num_classes))

            self.resize_param = 224

        elif self.model_name.lower() == "convnext":
            """ ConvNeXt small """
            self.net = models.convnext_small(weights='DEFAULT')

            self.input_features = self.net.classifier[2].in_features  # 768
            # self.net.classifier[2] = nn.Sequential(nn.Dropout(p=self.dropout),
            #                                        nn.Linear(input_features,
            #                                                  input_features // 2),
            #                                        nn.ReLU(inplace=True),
            #                                        nn.Dropout(p=self.dropout),
            #                                        nn.Linear(input_features // 2,
            #                                                  input_features // 4),
            #                                        nn.ReLU(inplace=True),
            #                                        nn.Dropout(p=self.dropout),
            #                                        nn.Linear(input_features // 4,
            #                                                  self.num_classes))

            self.resize_param = 224

        elif self.model_name.lower() == "swin":
            """ Swin Transformer V2 -T """
            self.net = models.swin_v2_t(
                weights=models.Swin_V2_T_Weights.DEFAULT)

            self.input_features = self.net.head.in_features  # 768
            # self.net.head = nn.Sequential(nn.Dropout(p=self.dropout),
            #                               nn.Linear(input_features,
            #                                         input_features // 2),
            #                               nn.ReLU(inplace=True),
            #                               nn.Dropout(p=self.dropout),
            #                               nn.Linear(input_features // 2,
            #                                         input_features // 4),
            #                               nn.ReLU(inplace=True),
            #                               nn.Dropout(p=self.dropout),
            #                               nn.Linear(input_features // 4,
            #                                         self.num_classes))

            self.resize_param = 224

        elif self.model_name.lower() == "efficient":
            """ EfficientNet b0 """
            self.net = models.efficientnet_b0(weights='DEFAULT')

            self.input_features = self.net.classifier[1].in_features  # 1200
            # self.net.classifier = nn.Sequential(nn.Dropout(p=self.dropout),
            #                                     nn.Linear(input_features,
            #                                               input_features // 2),
            #                                     nn.ReLU(inplace=True),
            #                                     nn.Dropout(p=self.dropout),
            #                                     nn.Linear(input_features // 2,
            #                                               input_features // 4),
            #                                     nn.ReLU(inplace=True),
            #                                     nn.Dropout(p=self.dropout),
            #                                     nn.Linear(input_features // 4,
            #                                               self.num_classes))

            self.resize_param = 224

        else:
            print("Invalid model name, MODEL NOT LOAD")
            TypeError(
                "Valid model names are 'resnet', 'convnext', 'swim' or 'efficient'"
            )
            exit()

    def set_parameter_requires_grad(model,
                                    number_frozen_layers,
                                    feature_layers=8):
        for k, child in enumerate(model.named_children()):
            if k == number_frozen_layers or k == feature_layers:
                break
            for param in child[1].parameters():
                param.requires_grad = False

        return model
