# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Emotion recognition model module (MobileNetV3-Large + SE).
Scale comparison experiment with larger backbone.
"""

# Version information
__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Face Emotion Recognition Model (Large SE)"

# Module imports
from mobilenetv3_large_se.config import ConfigLoader
from mobilenetv3_large_se.model import EmotionNet
from mobilenetv3_large_se.train import TrainModel
from mobilenetv3_large_se.predict import PredictEmotion
from mobilenetv3_large_se.utils import ModelHelper

# Export symbols
__all__ = [
	"ConfigLoader",
	"EmotionNet",
	"TrainModel",
	"PredictEmotion",
	"ModelHelper"
]