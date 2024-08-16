from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_layer(self, layer_name):
        """
        Method to retrieve a specific layer from the model.
        This is useful for Grad-CAM and other explainability methods.
        """
        pass
