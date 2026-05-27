# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Configuration file for evaluation module.
Defines evaluation parameters and path configurations.
"""

import os
from pathlib import Path


class EvalConfig:
	"""
	Evaluation configuration class.
	Provides centralized configuration for evaluation process.
	"""

	def __init__(self):
		"""Initialize evaluation configuration with default values."""
		self._dctConfig = {}
		self._LoadDefaults()

	def _LoadDefaults(self):
		"""Load default configuration values."""
		# Project root directory
		sCurrentDir = Path(__file__).parent
		self._sProjectRoot = sCurrentDir.parent

		# Build configuration dictionary
		self._dctConfig = {
			# Evaluation targets
			"fTargetAccuracy": 0.60,  # Target accuracy >= 60%
			"fMaxInferenceTime": 100.0,  # Max inference time <= 100ms
			"fMaxModelSize": 50.0,  # Max model size <= 50MB

			# Speed benchmark
			"iWarmupIterations": 10,
			"iBenchmarkIterations": 100,

			# Robustness test
			"lstLightingConditions": ["bright", "normal", "dark"],
			"lstPoseVariations": ["frontal", "slight_left", "slight_right"],
			"lstOcclusionTypes": ["none", "partial"],

			# Class names
			"lstEmotionLabels": [
				"angry", "disgust", "fear", "happy",
				"sad", "surprise", "neutral"
			],

			# Paths
			"sDataDir": str(self._sProjectRoot / "data_file" / "fer2013"),
			"sTestDir": str(self._sProjectRoot / "data_file" / "fer2013" / "test"),
			"sModelDir": str(self._sProjectRoot / "mobilenetv3_small_se" / "weights"),
			"sOutputDir": str(sCurrentDir / "outputs"),
			"sReportDir": str(sCurrentDir / "reports"),

			# Visualization
			"sFigureFormat": "png",
			"iFigureDpi": 150
		}

		# Create output directories
		self._CreateDirectories()

	def SetModelName(self, sModelName):
		"""
		Set model name and update output directories to model-specific subdirectories.

		Creates the following structure under evaluation/:
			outputs/{model_name}/
			reports/{model_name}/

		Args:
			sModelName (str): Model name (e.g., 'mobilenetv3_small_se').
		"""
		sCurrentDir = Path(__file__).parent
		self._dctConfig["sModelName"] = sModelName
		self._dctConfig["sOutputDir"] = str(sCurrentDir / "outputs" / sModelName)
		self._dctConfig["sReportDir"] = str(sCurrentDir / "reports" / sModelName)
		self._dctConfig["sModelDir"] = str(self._sProjectRoot / sModelName / "weights")
		self._CreateDirectories()

	def _CreateDirectories(self):
		"""Create necessary output directories."""
		for sKey in ["sOutputDir", "sReportDir"]:
			os.makedirs(self._dctConfig[sKey], exist_ok=True)

	def GetConfig(self):
		"""
		Get configuration dictionary.

		Returns:
			dict: Configuration dictionary.
		"""
		return self._dctConfig.copy()

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

	@property
	def TargetAccuracy(self):
		"""Get target accuracy."""
		return self._dctConfig["fTargetAccuracy"]

	@property
	def MaxInferenceTime(self):
		"""Get max inference time in ms."""
		return self._dctConfig["fMaxInferenceTime"]

	@property
	def MaxModelSize(self):
		"""Get max model size in MB."""
		return self._dctConfig["fMaxModelSize"]

	@property
	def EmotionLabels(self):
		"""Get emotion label list."""
		return self._dctConfig["lstEmotionLabels"]

	@property
	def ProjectRoot(self):
		"""Get project root path."""
		return self._sProjectRoot

	@property
	def OutputDir(self):
		"""Get output directory path."""
		return self._dctConfig["sOutputDir"]

	@property
	def ReportDir(self):
		"""Get report directory path."""
		return self._dctConfig["sReportDir"]


# Global configuration instance
_gConfig = EvalConfig()


def GetConfig():
	"""
	Get evaluation configuration instance.

	Returns:
		EvalConfig: Configuration instance.
	"""
	return _gConfig