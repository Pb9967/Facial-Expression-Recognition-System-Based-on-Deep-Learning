# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Model configuration file.
Defines hyperparameters and path configurations for emotion recognition model.
"""

import os
import torch
from pathlib import Path


class ConfigLoader:
	"""
	Configuration loader for emotion recognition model.
	Provides centralized configuration management.
	"""

	def __init__(self):
		"""Initialize configuration with default values."""
		self._dctConfig = {}
		self._LoadDefaults()

	def _LoadDefaults(self):
		"""Load default configuration values."""
		# Project root directory
		sCurrentDir = Path(__file__).parent
		sProjectRoot = sCurrentDir.parent

		# Dataset path
		sDataDir = sProjectRoot / "data_file" / "fer2013"
		sDataDir.mkdir(parents=True, exist_ok=True)

		# Build configuration dictionary
		self._dctConfig = {
			# Model configuration
			"sBackbone": "small",
			"bPretrained": True,
			"iNumClasses": 7,
			"tplInputSize": (224, 224),
			"fDropout": 0.2,

			# Training configuration
			"iEpochs": 50,
			"iBatchSize": 32,
			"fLearningRate": 2e-4,
			"fWeightDecay": 1e-4,
			"sOptimizer": "adam",
			"sScheduler": "cosine",
			"iGradientAccumSteps": 2,

			# Device configuration
			"sDevice": "cuda" if torch.cuda.is_available() else "cpu",
			"bUseAmp": True,

			# Data paths
			"sDataDir": str(sDataDir),
			"sTrainDir": str(sDataDir / "train"),
			"sValDir": str(sDataDir / "val"),
			"sTestDir": str(sDataDir / "test"),

			# Output paths (placeholder, will be overridden by train.py)
			"sOutputDir": str(sCurrentDir / "outputs"),
			"sWeightsDir": str(sCurrentDir / "weights"),
			"sLogsDir": str(sCurrentDir / "logs"),

			# Pretrained weights
			"sPretrainedPath": None,

			# Data loading
			"iNumWorkers": 4,
			"bPinMemory": False,
			"bShuffle": True,
			"bPersistentWorkers": True,
			"iPrefetchFactor": 2,

			# Model saving
			"bSaveBest": True,
			"bSaveFinal": True
		}

	def CreateOutputDirs(self):
		"""
		Create output directories based on current config.
		To be called by train.py after paths are finalized.
		"""
		for sKey in ["sOutputDir", "sWeightsDir", "sLogsDir"]:
			os.makedirs(self._dctConfig[sKey], exist_ok=True)

	def GetConfig(self):
		"""
		Get configuration dictionary.

		Returns:
			dict: Configuration dictionary.
		"""
		return self._dctConfig.copy()

	def UpdateConfig(self, dctUpdates):
		"""
		Update configuration with new values.

		Args:
			dctUpdates (dict): Dictionary of updates.
		"""
		self._dctConfig.update(dctUpdates)

	def Get(self, sKey, DefaultValue=None):
		"""
		Get specific configuration value.

		Args:
			sKey (str): Configuration key.
			DefaultValue: Default value if key not found.

		Returns:
			Configuration value.
		"""
		return self._dctConfig.get(sKey, DefaultValue)


def GetConfig():
	"""
	Get model configuration.

	Returns:
		dict: Model configuration dictionary.
	"""
	loader = ConfigLoader()
	return loader.GetConfig()