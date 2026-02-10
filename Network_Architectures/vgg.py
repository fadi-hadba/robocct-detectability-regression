import torch
import torch.nn as nn


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(VGGBlock, self).__init__()
        layers = []
        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGGRegression(nn.Module):
    def __init__(self, vgg_block, num_regression_targets=1):
        super(VGGRegression, self).__init__()
        self.features = nn.Sequential(
            vgg_block(1, 64, 2),  # Single input channel
            vgg_block(64, 128, 2),
            vgg_block(128, 256, 3),
            vgg_block(256, 512, 3),
            vgg_block(512, 512, 3)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_regression_targets), # Regression output

        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def VGG16(num_regression_targets=1):
    return VGGRegression(VGGBlock, num_regression_targets)


def VGG19(num_regression_targets=1):
    return VGGRegression(VGGBlock, num_regression_targets)