# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Emotion recognition model architecture.
Based on MobileNetV3-Large backbone with SE Block attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.ops import SqueezeExcitation


class EmotionNet(nn.Module):
	"""
	Emotion recognition network.
	Based on MobileNetV3-Large backbone with SE Block attention mechanism.
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

		# Build SE attention block (public attribute for weight loading)
		self.se_block = self._CreateSEBlock()

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

	def _CreateSEBlock(self):
		"""
		Create SE attention block with correct channel dimensions.

		Returns:
			nn.Module: SE Block module.
		"""
		# Large backbone feature channels
		iInputChannels = 960

		# Create SE block for channel attention
		return SqueezeExcitation(
			input_channels=iInputChannels,
			squeeze_channels=64
		)

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

		# Apply SE attention
		x = self.se_block(x)

		# Global average pooling
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
