# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Confusion matrix visualization for emotion recognition.
Generates heatmap visualization of classification results.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from evaluation.config import GetConfig
from evaluation.utils import GetTimestamp


class ConfusionMatrixPlotter:
	"""
	Generates and saves confusion matrix visualizations.
	Supports both normalized and raw count matrices.
	"""

	def __init__(self):
		"""Initialize confusion matrix plotter."""
		self._oConfig = GetConfig()
		self._lstEmotionLabels = self._oConfig.EmotionLabels
		self._sOutputDir = self._oConfig.OutputDir

	def Plot(self, arrMatrix, sTitle="Confusion Matrix", bNormalize=True, sSavePath=None):
		"""
		Plot confusion matrix heatmap.

		Args:
			arrMatrix (np.ndarray): Confusion matrix data.
			sTitle (str): Plot title.
			bNormalize (bool): Whether to normalize the matrix.
			sSavePath (str): Path to save the figure. If None, auto-generates.

		Returns:
			str: Path to saved figure.
		"""
		# Normalize if requested
		if bNormalize:
			arrPlotMatrix = self._NormalizeMatrix(arrMatrix)
			sValueFormat = ".2f"
			sTitle += " (Normalized)"
		else:
			arrPlotMatrix = arrMatrix
			sValueFormat = "d"

		# Create figure
		fig, ax = plt.subplots(figsize=(10, 8))

		# Plot heatmap
		sns.heatmap(
			arrPlotMatrix,
			annot=True,
			fmt=sValueFormat,
			cmap="Blues",
			xticklabels=self._lstEmotionLabels,
			yticklabels=self._lstEmotionLabels,
			ax=ax,
			cbar_kws={"label": "Proportion" if bNormalize else "Count"}
		)

		# Configure plot
		ax.set_xlabel("Predicted Label", fontsize=12)
		ax.set_ylabel("True Label", fontsize=12)
		ax.set_title(sTitle, fontsize=14, fontweight="bold")
		ax.tick_params(axis="both", labelsize=10)

		# Rotate x-axis labels
		plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

		# Adjust layout
		plt.tight_layout()

		# Save figure
		if sSavePath is None:
			sSavePath = os.path.join(
				self._sOutputDir,
				f"confusion_matrix_{GetTimestamp()}.png"
			)

		fig.savefig(sSavePath, dpi=self._oConfig.Get("iFigureDpi"), bbox_inches="tight")
		plt.close(fig)

		return sSavePath

	def PlotBoth(self, arrMatrix, sTitle="Confusion Matrix", sSavePath=None):
		"""
		Plot both normalized and raw confusion matrices side by side.

		Args:
			arrMatrix (np.ndarray): Confusion matrix data.
			sTitle (str): Plot title prefix.
			sSavePath (str): Path to save the figure.

		Returns:
			str: Path to saved figure.
		"""
		# Create figure with two subplots
		fig, axes = plt.subplots(1, 2, figsize=(16, 7))

		# Left: Raw counts
		sns.heatmap(
			arrMatrix,
			annot=True,
			fmt="d",
			cmap="Blues",
			xticklabels=self._lstEmotionLabels,
			yticklabels=self._lstEmotionLabels,
			ax=axes[0],
			cbar_kws={"label": "Count"}
		)
		axes[0].set_xlabel("Predicted Label", fontsize=11)
		axes[0].set_ylabel("True Label", fontsize=11)
		axes[0].set_title(f"{sTitle} (Counts)", fontsize=13, fontweight="bold")
		axes[0].tick_params(axis="both", labelsize=9)

		# Right: Normalized
		arrNormalized = self._NormalizeMatrix(arrMatrix)
		sns.heatmap(
			arrNormalized,
			annot=True,
			fmt=".2f",
			cmap="Blues",
			xticklabels=self._lstEmotionLabels,
			yticklabels=self._lstEmotionLabels,
			ax=axes[1],
			cbar_kws={"label": "Proportion"}
		)
		axes[1].set_xlabel("Predicted Label", fontsize=11)
		axes[1].set_ylabel("True Label", fontsize=11)
		axes[1].set_title(f"{sTitle} (Normalized)", fontsize=13, fontweight="bold")
		axes[1].tick_params(axis="both", labelsize=9)

		# Rotate x-axis labels
		for ax in axes:
			plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

		# Adjust layout
		plt.tight_layout()

		# Save figure
		if sSavePath is None:
			sSavePath = os.path.join(
				self._sOutputDir,
				f"confusion_matrix_both_{GetTimestamp()}.png"
			)

		fig.savefig(sSavePath, dpi=self._oConfig.Get("iFigureDpi"), bbox_inches="tight")
		plt.close(fig)

		return sSavePath

	def _NormalizeMatrix(self, arrMatrix):
		"""
		Normalize confusion matrix by row (true labels).

		Args:
			arrMatrix (np.ndarray): Raw confusion matrix.

		Returns:
			np.ndarray: Normalized confusion matrix.
		"""
		arrRowSums = arrMatrix.sum(axis=1, keepdims=True)
		# Avoid division by zero
		arrRowSums = np.where(arrRowSums == 0, 1, arrRowSums)
		return arrMatrix.astype("float") / arrRowSums

	def AnalyzeErrors(self, arrMatrix):
		"""
		Analyze common classification errors from confusion matrix.

		Args:
			arrMatrix (np.ndarray): Confusion matrix data.

		Returns:
			dict: Error analysis results.
		"""
		dctErrors = {}
		arrNormalized = self._NormalizeMatrix(arrMatrix)

		# Find top confusions
		lstConfusions = []
		for i in range(len(self._lstEmotionLabels)):
			for j in range(len(self._lstEmotionLabels)):
				if i != j and arrNormalized[i, j] > 0.05:  # More than 5% confusion
					lstConfusions.append({
						"true_label": self._lstEmotionLabels[i],
						"predicted_label": self._lstEmotionLabels[j],
						"rate": float(arrNormalized[i, j]),
						"count": int(arrMatrix[i, j])
					})

		# Sort by confusion rate
		lstConfusions.sort(key=lambda x: x["rate"], reverse=True)

		dctErrors["top_confusions"] = lstConfusions[:10]
		dctErrors["most_confused_class"] = self._GetMostConfusedClass(arrNormalized)

		return dctErrors

	def _GetMostConfusedClass(self, arrNormalized):
		"""
		Find the most confused class (highest misclassification rate).

		Args:
			arrNormalized (np.ndarray): Normalized confusion matrix.

		Returns:
			dict: Most confused class information.
		"""
		lstMisclassRates = []
		for i in range(len(self._lstEmotionLabels)):
			# Misclassification rate = 1 - true positive rate
			fMisclassRate = 1.0 - arrNormalized[i, i]
			lstMisclassRates.append({
				"class": self._lstEmotionLabels[i],
				"misclassification_rate": float(fMisclassRate)
			})

		lstMisclassRates.sort(key=lambda x: x["misclassification_rate"], reverse=True)
		return lstMisclassRates[0]
