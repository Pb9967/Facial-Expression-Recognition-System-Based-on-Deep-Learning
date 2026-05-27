# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Emotion recognition model architecture.
Based on MobileNetV3-Large backbone (NO SE Block).
Ablation baseline: remove SE Block attention mechanism from large backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights


class EmotionNet(nn.Module):
    """
    Emotion recognition network.
    Based on MobileNetV3-Large backbone (NO SE Block).
    """

    # Emotion class labels (class constant)
    CLST_EMOTION_LABELS = [
        "angry",      # Anger
        "disgust",    # Disgust
        "fear",       # Fear
        "happy",      # Happiness
        "sad",        # Sadness
        "surprise",   # Surprise
        "neutral"     # Neutral
    ]

    def __init__(self, dctConfig=None, bPretrained=True):
        """
        Initialize emotion recognition model.

        Args:
            dctConfig (dict): Model configuration dictionary.
            bPretrained (bool): Whether to use pretrained weights.
        """
        super(EmotionNet, self).__init__()

        # Store configuration (private attribute, not saved to state_dict)
        self._dctConfig = dctConfig if dctConfig else {"sBackbone": "large"}

        # Build backbone network (public attribute for weight loading)
        self.backbone = self._CreateBackbone(bPretrained)

        # Build classifier head (public attribute for weight loading)
        self.classifier = self._CreateClassifier()

    def _CreateBackbone(self, bPretrained):
        """
        Create backbone network based on configuration.

        Args:
            bPretrained (bool): Whether to use pretrained weights.

        Returns:
            nn.Module: Backbone network.
        """
        weights = MobileNet_V3_Large_Weights.DEFAULT if bPretrained else None
        backbone = mobilenet_v3_large(weights=weights)

        # Remove original classifier
        del backbone.classifier

        return backbone

    def _CreateClassifier(self):
        """
        Create classifier head with dropout.

        Returns:
            nn.Sequential: Classifier module.
        """
        # Large backbone feature dimension
        iFeatureDim = 960

        # Get dropout rate from config
        fDropout = self._dctConfig.get("fDropout", 0.2)

        # Build classifier with Hardswish activation
        return nn.Sequential(
            nn.Linear(iFeatureDim, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(fDropout),
            nn.Linear(256, len(self.CLST_EMOTION_LABELS))
        )

    def forward(self, x, bReturnLogits=True):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input image tensor.
            bReturnLogits (bool): Return logits if True, probabilities if False.

        Returns:
            torch.Tensor: Output logits or probabilities.
        """
        # Extract features through backbone
        x = self.backbone.features(x)

        # Global average pooling (NO SE Block in ablation baseline)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        # Apply softmax only during inference when probabilities needed
        if not self.training and not bReturnLogits:
            x = F.softmax(x, dim=1)

        return x

    def GetEmotionLabels(self):
        """
        Get emotion class labels.

        Returns:
            list: List of emotion label strings.
        """
        return self.CLST_EMOTION_LABELS.copy()

    def GetNumClasses(self):
        """
        Get number of emotion classes.

        Returns:
            int: Number of classes.
        """
        return len(self.CLST_EMOTION_LABELS)


def CreateModel(dctConfig=None, bPretrained=True):
    """
    Create emotion recognition model instance.

    Args:
        dctConfig (dict): Model configuration dictionary.
        bPretrained (bool): Whether to use pretrained weights.

    Returns:
        EmotionNet: Initialized model instance.
    """
    model = EmotionNet(dctConfig, bPretrained)
    return model
