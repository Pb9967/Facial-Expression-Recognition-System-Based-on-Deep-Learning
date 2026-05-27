# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Emotion recognition model module.
Provides MobileNetV3 + SE Block based face emotion recognition.
Supports training, validation and inference.
"""

# Version information
__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Face Emotion Recognition Model"

# Module imports
from mobilenetv3_small_se.config import ConfigLoader
from mobilenetv3_small_se.model import EmotionNet
from mobilenetv3_small_se.train import TrainModel
from mobilenetv3_small_se.predict import PredictEmotion
from mobilenetv3_small_se.utils import ModelHelper

# Export symbols
__all__ = [
	"ConfigLoader",
	"EmotionNet",
	"TrainModel",
	"PredictEmotion",
	"ModelHelper"
]