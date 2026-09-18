"""An explicit logistic clinical baseline over fitted numeric/one-hot features."""

from torch import nn


class ClinicalClassifier(nn.Module):
    def __init__(self, dimensions, num_classes):
        super().__init__()
        self.classifier = nn.Linear(dimensions, num_classes)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features):
        return self.classifier(features)
