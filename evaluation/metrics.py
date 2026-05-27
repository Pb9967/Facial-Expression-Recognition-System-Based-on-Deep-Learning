# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Metrics calculator for emotion recognition evaluation.
Computes accuracy, precision, recall, and F1-score.
"""

import numpy as np
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	classification_report
)

from evaluation.config import GetConfig
from evaluation.utils import EnsureNumpy


class MetricsCalculator:
	"""
	Calculates evaluation metrics for emotion recognition model.
	Supports per-class and overall metrics computation.
	"""

	def __init__(self):
		"""Initialize metrics calculator."""
		self._oConfig = GetConfig()
		self._lstEmotionLabels = self._oConfig.EmotionLabels
		self._lstYPred = []
		self._lstYTrue = []

	def AddBatch(self, yPred, yTrue):
		"""
		Add a batch of predictions and ground truth labels.

		Args:
			yPred: Predicted labels (tensor or numpy array).
			yTrue: Ground truth labels (tensor or numpy array).
		"""
		yPred = EnsureNumpy(yPred)
		yTrue = EnsureNumpy(yTrue)

		self._lstYPred.extend(yPred.flatten().tolist())
		self._lstYTrue.extend(yTrue.flatten().tolist())

	def Reset(self):
		"""Reset accumulated predictions and labels."""
		self._lstYPred = []
		self._lstYTrue = []

	def Compute(self):
		"""
		Compute all evaluation metrics.

		Returns:
			dict: Dictionary containing all metrics.
		"""
		if not self._lstYPred or not self._lstYTrue:
			return {}

		arrYPred = np.array(self._lstYPred)
		arrYTrue = np.array(self._lstYTrue)

		# Overall metrics
		dctMetrics = {
			"accuracy": float(accuracy_score(arrYTrue, arrYPred)),
			"precision_macro": float(precision_score(
				arrYTrue, arrYPred, average="macro", zero_division=0
			)),
			"recall_macro": float(recall_score(
				arrYTrue, arrYPred, average="macro", zero_division=0
			)),
			"f1_macro": float(f1_score(
				arrYTrue, arrYPred, average="macro", zero_division=0
			)),
			"precision_weighted": float(precision_score(
				arrYTrue, arrYPred, average="weighted", zero_division=0
			)),
			"recall_weighted": float(recall_score(
				arrYTrue, arrYPred, average="weighted", zero_division=0
			)),
			"f1_weighted": float(f1_score(
				arrYTrue, arrYPred, average="weighted", zero_division=0
			))
		}

		# Per-class metrics
		dctMetrics["per_class"] = self._ComputePerClass(arrYTrue, arrYPred)

		# Check if target accuracy is met
		dctMetrics["target_met"] = dctMetrics["accuracy"] >= self._oConfig.TargetAccuracy

		return dctMetrics

	def _ComputePerClass(self, arrYTrue, arrYPred):
		"""
		Compute per-class metrics.

		Args:
			arrYTrue (np.ndarray): Ground truth labels.
			arrYPred (np.ndarray): Predicted labels.

		Returns:
			dict: Per-class metrics dictionary.
		"""
		dctPerClass = {}

		for iIdx, sLabel in enumerate(self._lstEmotionLabels):
			# Binary classification for each class
			arrBinaryTrue = (arrYTrue == iIdx).astype(int)
			arrBinaryPred = (arrYPred == iIdx).astype(int)

			dctPerClass[sLabel] = {
				"precision": float(precision_score(
					arrBinaryTrue, arrBinaryPred, zero_division=0
				)),
				"recall": float(recall_score(
					arrBinaryTrue, arrBinaryPred, zero_division=0
				)),
				"f1": float(f1_score(
					arrBinaryTrue, arrBinaryPred, zero_division=0
				)),
				"support": int(arrBinaryTrue.sum())
			}

		return dctPerClass

	def GetClassificationReport(self):
		"""
		Get sklearn classification report string.

		Returns:
			str: Classification report.
		"""
		if not self._lstYPred or not self._lstYTrue:
			return ""

		return classification_report(
			self._lstYTrue,
			self._lstYPred,
			target_names=self._lstEmotionLabels,
			zero_division=0
		)

	def GetConfusionMatrix(self):
		"""
		Get confusion matrix data.

		Returns:
			np.ndarray: Confusion matrix.
		"""
		if not self._lstYPred or not self._lstYTrue:
			return np.zeros((len(self._lstEmotionLabels), len(self._lstEmotionLabels)))

		from sklearn.metrics import confusion_matrix

		return confusion_matrix(self._lstYTrue, self._lstYPred)
