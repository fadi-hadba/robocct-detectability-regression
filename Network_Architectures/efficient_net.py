import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetRegression(nn.Module):
    def __init__(self, backbone, num_regression_targets, num_input_channels=1):
        super(EfficientNetRegression, self).__init__()
        self.model = EfficientNet.from_pretrained(backbone)
        if backbone in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2']:
            self.model._conv_stem = nn.Conv2d(num_input_channels, 32, kernel_size=3, stride=2, bias=False)  # Anpassung der ersten Convolutional-Schicht
        elif backbone in ['efficientnet-b3']:
            self.model._conv_stem = nn.Conv2d(num_input_channels, 40, kernel_size=3, stride=2,bias=False)  # Anpassung der ersten Convolutional-Schicht
        elif backbone in ['efficientnet-b4', 'efficientnet-b5']:
            self.model._conv_stem = nn.Conv2d(num_input_channels, 48, kernel_size=3, stride=2,bias=False)  # Anpassung der ersten Convolutional-Schicht
        elif backbone in ['efficientnet-b6']:
            self.model._conv_stem = nn.Conv2d(num_input_channels, 56, kernel_size=3, stride=2,bias=False)  # Anpassung der ersten Convolutional-Schicht
        elif backbone in ['efficientnet-b7']:
            self.model._conv_stem = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=2,bias=False)  # Anpassung der ersten Convolutional-Schicht

        self.model._fc = nn.Linear(self.model._fc.in_features, num_regression_targets)

    def forward(self, x):
        return self.model(x)


def EfficientNetB0(num_regression_targets=1):
    return EfficientNetRegression(backbone='efficientnet-b0', num_regression_targets=num_regression_targets, num_input_channels=1)

def EfficientNetB1(num_regression_targets=1):
    return EfficientNetRegression(backbone='efficientnet-b1', num_regression_targets=num_regression_targets, num_input_channels=1)

def EfficientNetB2(num_regression_targets=1):
    return EfficientNetRegression(backbone='efficientnet-b2', num_regression_targets=num_regression_targets, num_input_channels=1)

def EfficientNetB3(num_regression_targets=1):
    return EfficientNetRegression(backbone='efficientnet-b3', num_regression_targets=num_regression_targets, num_input_channels=1)

def EfficientNetB4(num_regression_targets=1):
    return EfficientNetRegression(backbone='efficientnet-b4', num_regression_targets=num_regression_targets, num_input_channels=1)

def EfficientNetB5(num_regression_targets=1):
    return EfficientNetRegression(backbone='efficientnet-b5', num_regression_targets=num_regression_targets, num_input_channels=1)

def EfficientNetB6(num_regression_targets=1):
    return EfficientNetRegression(backbone='efficientnet-b6', num_regression_targets=num_regression_targets, num_input_channels=1)

def EfficientNetB7(num_regression_targets=1):
    return EfficientNetRegression(backbone='efficientnet-b7', num_regression_targets=num_regression_targets, num_input_channels=1)

