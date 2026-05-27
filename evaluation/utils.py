# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Utility functions for evaluation module.
Provides helper functions for data processing and visualization.
"""

import json
import time
from datetime import datetime

import numpy as np
import torch


def FormatTime(fSeconds):
	"""
	Format seconds to readable time string.

	Args:
		fSeconds (float): Time in seconds.

	Returns:
		str: Formatted time string.
	"""
	if fSeconds < 1:
		return f"{fSeconds * 1000:.2f}ms"
	elif fSeconds < 60:
		return f"{fSeconds:.2f}s"
	else:
		iMinutes = int(fSeconds // 60)
		fSecs = fSeconds % 60
		return f"{iMinutes}m {fSecs:.2f}s"


def FormatSize(iBytes):
	"""
	Format bytes to readable size string.

	Args:
		iBytes (int): Size in bytes.

	Returns:
		str: Formatted size string.
	"""
	for sUnit in ["B", "KB", "MB", "GB"]:
		if iBytes < 1024.0:
			return f"{iBytes:.2f}{sUnit}"
		iBytes /= 1024.0
	return f"{iBytes:.2f}TB"


def GetModelSize(sModelPath):
	"""
	Get model file size in MB.

	Args:
		sModelPath (str): Path to model file.

	Returns:
		float: Model size in MB.
	"""
	import os

	if not os.path.exists(sModelPath):
		return 0.0

	iSizeBytes = os.path.getsize(sModelPath)
	return iSizeBytes / (1024 * 1024)


def CountParameters(model):
	"""
	Count trainable parameters in model.

	Args:
		model (nn.Module): PyTorch model.

	Returns:
		int: Number of trainable parameters.
	"""
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def EnsureNumpy(data):
	"""
	Convert tensor to numpy array if needed.

	Args:
		data: Input data (tensor or numpy array).

	Returns:
		np.ndarray: Numpy array.
	"""
	if isinstance(data, torch.Tensor):
		return data.detach().cpu().numpy()
	return np.array(data)


def SaveJson(data, sPath):
	"""
	Save dictionary to JSON file.

	Args:
		data (dict): Data to save.
		sPath (str): Output file path.
	"""
	with open(sPath, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=False)


def LoadJson(sPath):
	"""
	Load dictionary from JSON file.

	Args:
		sPath (str): Input file path.

	Returns:
		dict: Loaded data.
	"""
	with open(sPath, "r", encoding="utf-8") as f:
		return json.load(f)


def GetTimestamp():
	"""
	Get current timestamp string.

	Returns:
		str: Formatted timestamp.
	"""
	return datetime.now().strftime("%Y%m%d_%H%M%S")


class Timer:
	"""Simple timer for measuring execution time."""

	def __init__(self):
		"""Initialize timer."""
		self._fStartTime = None
		self._fEndTime = None

	def Start(self):
		"""Start the timer."""
		self._fStartTime = time.time()
		return self

	def Stop(self):
		"""Stop the timer."""
		self._fEndTime = time.time()
		return self

	def Elapsed(self):
		"""
		Get elapsed time in seconds.

		Returns:
			float: Elapsed time.
		"""
		if self._fEndTime is None:
			return time.time() - self._fStartTime
		return self._fEndTime - self._fStartTime

	def __enter__(self):
		"""Context manager entry."""
		self.Start()
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Context manager exit."""
		self.Stop()
