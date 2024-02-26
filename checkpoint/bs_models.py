import torchvision.models
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights, ViT_B_16_Weights, Swin_T_Weights


class mobilenet_v3(nn.Module):
    def __init__(self, num_classes1):
        """
        MobileNetV3
        :param num_classes1: output dimension
        """
        super(mobilenet_v3, self).__init__()
        self.backbone = torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = 576
        self.backbone.classifier = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)

    def forward(self, x):
        """
        forward pass of mobilenet_v3
        :param x: the provided input tensor
        :return: the current position
        """
        x = self.backbone(x)
        return self.head_label(x)


class resnet18(nn.Module):
    def __init__(self, num_classes1):
        """
        ResNet18
        :param num_classes1: output dimension
        """
        super(resnet18, self).__init__()
        self.backbone = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)

    def forward(self, x):
        """
        forward pass of resnet18
        :param x: the provided input tensor
        :return: the current position
        """
        x = self.backbone(x)
        return self.head_label(x)


class vit(nn.Module):
    def __init__(self, num_classes1):
        """
        Vision Transformer b-16
        :param num_classes1: output dimension
        """
        super(vit, self).__init__()
        self.backbone = torchvision.models.vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = 384
        self.backbone.head = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)

    def forward(self, x):
        """
        forward pass of vit
        :param x: the provided input tensor
        :return: the current position
        """
        x = self.backbone(x)
        return self.head_label(x)


class swint(nn.Module):
    def __init__(self, num_classes1):
        """
        Swin Transformer tiny
        :param num_classes1: output dimension
        """
        super(swint, self).__init__()
        self.backbone = torchvision.models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)

    def forward(self, x):
        """
        forward pass of swint
        :param x: the provided input tensor
        :return: the current position
        """
        x = self.backbone(x)
        return self.head_label(x)
