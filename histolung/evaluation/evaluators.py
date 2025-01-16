from abc import ABC, abstractmethod

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch


class BaseEvaluator(ABC):

    def __init__(self, test_loader):
        self.test_loader = test_loader

    @abstractmethod
    def evaluate(self, model):
        pass


class KNNEvaluator(BaseEvaluator):

    def __init__(self, test_loader, n_neighbors=5, weights='uniform'):
        super().__init__(test_loader)
        self.n_neighbors = n_neighbors
        self.weights = weights

    def evaluate(self, model):
        features, labels = self.extract_features(model)
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                   weights=self.weights)
        knn.fit(features, labels)
        predictions = knn.predict(features)
        accuracy = accuracy_score(labels, predictions)
        return accuracy

    def extract_features(self, model):
        features, labels = [], []
        model.eval()
        with torch.no_grad():
            for images, targets in self.test_loader:
                features.append(model(images.to(model.device)).cpu().numpy())
                labels.append(targets.numpy())
        return np.vstack(features), np.concatenate(labels)


class LinearEvaluator(BaseEvaluator):

    def __init__(self, test_loader, max_iter=1000):
        super().__init__(test_loader)
        self.max_iter = max_iter

    def evaluate(self, model):
        features, labels = self.extract_features(model)
        clf = LogisticRegression(max_iter=self.max_iter)
        clf.fit(features, labels)
        predictions = clf.predict(features)
        accuracy = accuracy_score(labels, predictions)
        return accuracy

    def extract_features(self, model):
        features, labels = [], []
        model.eval()
        with torch.no_grad():
            for images, targets in self.test_loader:
                features.append(model(images.to(model.device)).cpu().numpy())
                labels.append(targets.numpy())
        return np.vstack(features), np.concatenate(labels)


def calculate_iou():
    pass


def calculate_dice():
    pass


class SegmentationEvaluator(BaseEvaluator):

    def evaluate(self, model):
        iou_scores, dice_scores = [], []
        model.eval()
        with torch.no_grad():
            for images, masks in self.test_loader:
                preds = model(images.to(model.device)).cpu()
                preds = (preds > 0.5).float()
                iou_scores.append(calculate_iou(preds, masks))
                dice_scores.append(calculate_dice(preds, masks))
        return np.mean(iou_scores), np.mean(dice_scores)
