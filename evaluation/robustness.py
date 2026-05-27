# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Robustness analyzer for emotion recognition model.
Tests model performance under various challenging conditions.
"""

import os
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from evaluation.config import GetConfig
from evaluation.metrics import MetricsCalculator
from evaluation.utils import EnsureNumpy


class RobustnessAnalyzer:
	"""
	Analyzes model robustness under various conditions.
	Tests lighting, pose, occlusion, and noise robustness.
	"""

	def __init__(self, model=None, sDevice="cuda"):
		"""
		Initialize robustness analyzer.

		Args:
			model (nn.Module): PyTorch model to analyze.
			sDevice (str): Device to run analysis on.
		"""
		self._oConfig = GetConfig()
		self._oModel = model
		self._sDevice = sDevice
		self._tplInputSize = (224, 224)
		self._oMetrics = MetricsCalculator()
		self._oTransform = None

	def SetModel(self, model, transform=None):
		"""
		Set model and transform for analysis.

		Args:
			model (nn.Module): PyTorch model.
			transform: Image transform pipeline.
		"""
		self._oModel = model
		self._oTransform = transform

	def Analyze(self, arrImages, arrLabels):
		"""
		Run comprehensive robustness analysis.

		Args:
			arrImages (np.ndarray): Test images array.
			arrLabels (np.ndarray): True labels array.

		Returns:
			dict: Robustness analysis results.
		"""
		if self._oModel is None:
			raise ValueError("Model not set. Call SetModel() first.")

		self._oModel.eval()
		self._oModel.to(self._sDevice)

		dctResults = {}

		# Baseline performance
		dctResults["baseline"] = self._EvaluateBaseline(arrImages, arrLabels)

		# Lighting robustness
		dctResults["lighting"] = self._TestLightingRobustness(arrImages, arrLabels)

		# Noise robustness
		dctResults["noise"] = self._TestNoiseRobustness(arrImages, arrLabels)

		# Blur robustness
		dctResults["blur"] = self._TestBlurRobustness(arrImages, arrLabels)

		# Contrast robustness
		dctResults["contrast"] = self._TestContrastRobustness(arrImages, arrLabels)

		# Overall robustness score
		dctResults["overall_score"] = self._CalculateOverallScore(dctResults)

		return dctResults

	def _EvaluateBaseline(self, arrImages, arrLabels):
		"""
		Evaluate baseline performance without perturbations.

		Args:
			arrImages: Test images.
			arrLabels: True labels.

		Returns:
			dict: Baseline metrics.
		"""
		self._oMetrics.Reset()
		self._EvaluateBatch(arrImages, arrLabels)
		return self._oMetrics.Compute()

	def _TestLightingRobustness(self, arrImages, arrLabels):
		"""
		Test model robustness to lighting variations.

		Args:
			arrImages: Test images.
			arrLabels: True labels.

		Returns:
			dict: Lighting robustness results.
		"""
		lstFactors = [0.3, 0.5, 0.7, 1.3, 1.5, 1.8]
		lstResults = []

		for fFactor in lstFactors:
			# Apply brightness change
			arrModified = self._AdjustBrightness(arrImages, fFactor)

			# Evaluate
			self._oMetrics.Reset()
			self._EvaluateBatch(arrModified, arrLabels)
			dctMetrics = self._oMetrics.Compute()

			lstResults.append({
				"brightness_factor": fFactor,
				"accuracy": dctMetrics.get("accuracy", 0),
				"f1_score": dctMetrics.get("f1_macro", 0)
			})

		return lstResults

	def _TestNoiseRobustness(self, arrImages, arrLabels):
		"""
		Test model robustness to Gaussian noise.

		Args:
			arrImages: Test images.
			arrLabels: True labels.

		Returns:
			dict: Noise robustness results.
		"""
		lstNoiseLevels = [0.01, 0.02, 0.05, 0.1, 0.15]
		lstResults = []

		for fNoiseStd in lstNoiseLevels:
			# Add Gaussian noise
			arrModified = self._AddGaussianNoise(arrImages, fNoiseStd)

			# Evaluate
			self._oMetrics.Reset()
			self._EvaluateBatch(arrModified, arrLabels)
			dctMetrics = self._oMetrics.Compute()

			lstResults.append({
				"noise_std": fNoiseStd,
				"accuracy": dctMetrics.get("accuracy", 0),
				"f1_score": dctMetrics.get("f1_macro", 0)
			})

		return lstResults

	def _TestBlurRobustness(self, arrImages, arrLabels):
		"""
		Test model robustness to image blur.

		Args:
			arrImages: Test images.
			arrLabels: True labels.

		Returns:
			dict: Blur robustness results.
		"""
		lstBlurRadii = [1, 2, 3, 5, 7]
		lstResults = []

		for iRadius in lstBlurRadii:
			# Apply Gaussian blur
			arrModified = self._ApplyBlur(arrImages, iRadius)

			# Evaluate
			self._oMetrics.Reset()
			self._EvaluateBatch(arrModified, arrLabels)
			dctMetrics = self._oMetrics.Compute()

			lstResults.append({
				"blur_radius": iRadius,
				"accuracy": dctMetrics.get("accuracy", 0),
				"f1_score": dctMetrics.get("f1_macro", 0)
			})

		return lstResults

	def _TestContrastRobustness(self, arrImages, arrLabels):
		"""
		Test model robustness to contrast variations.

		Args:
			arrImages: Test images.
			arrLabels: True labels.

		Returns:
			dict: Contrast robustness results.
		"""
		lstContrastFactors = [0.3, 0.5, 0.7, 1.3, 1.5, 2.0]
		lstResults = []

		for fFactor in lstContrastFactors:
			# Apply contrast change
			arrModified = self._AdjustContrast(arrImages, fFactor)

			# Evaluate
			self._oMetrics.Reset()
			self._EvaluateBatch(arrModified, arrLabels)
			dctMetrics = self._oMetrics.Compute()

			lstResults.append({
				"contrast_factor": fFactor,
				"accuracy": dctMetrics.get("accuracy", 0),
				"f1_score": dctMetrics.get("f1_macro", 0)
			})

		return lstResults

	def _EvaluateBatch(self, arrImages, arrLabels, iBatchSize=32):
		"""
		Evaluate images in batches.

		Args:
			arrImages: Images to evaluate.
			arrLabels: True labels.
			iBatchSize (int): Batch size.
		"""
		self._oModel.eval()
		iNumSamples = len(arrImages)

		with torch.no_grad():
			for iStart in range(0, iNumSamples, iBatchSize):
				iEnd = min(iStart + iBatchSize, iNumSamples)
				arrBatchImages = arrImages[iStart:iEnd]
				arrBatchLabels = arrLabels[iStart:iEnd]

				# Convert to tensor
				if self._oTransform:
					lstTensors = []
					for img in arrBatchImages:
						if isinstance(img, np.ndarray):
							oPilImg = Image.fromarray(img.astype("uint8"))
						else:
							oPilImg = img
						lstTensors.append(self._oTransform(oPilImg))
					tensorBatch = torch.stack(lstTensors).to(self._sDevice)
				else:
					tensorBatch = torch.from_numpy(arrBatchImages).float().to(self._sDevice)
					if tensorBatch.dim() == 3:
						tensorBatch = tensorBatch.unsqueeze(1)
					tensorBatch = tensorBatch.permute(0, 3, 1, 2)

				# Inference
				tensorOutput = self._oModel(tensorBatch)
				tensorPred = torch.argmax(tensorOutput, dim=1)

				# Add to metrics
				self._oMetrics.AddBatch(tensorPred.cpu().numpy(), arrBatchLabels)

	# Perturbation functions
	def _AdjustBrightness(self, arrImages, fFactor):
		"""Adjust image brightness."""
		arrResult = arrImages.copy().astype("float32")
		arrResult = np.clip(arrResult * fFactor, 0, 255).astype("uint8")
		return arrResult

	def _AddGaussianNoise(self, arrImages, fStd):
		"""Add Gaussian noise to images."""
		arrResult = arrImages.copy().astype("float32")
		arrNoise = np.random.normal(0, fStd * 255, arrResult.shape)
		arrResult = np.clip(arrResult + arrNoise, 0, 255).astype("uint8")
		return arrResult

	def _ApplyBlur(self, arrImages, iRadius):
		"""Apply Gaussian blur to images."""
		lstResult = []
		for img in arrImages:
			oPilImg = Image.fromarray(img if isinstance(img, np.ndarray) else np.array(img))
			oBlurred = oPilImg.filter(ImageFilter.GaussianBlur(radius=iRadius))
			lstResult.append(np.array(oBlurred))
		return np.array(lstResult)

	def _AdjustContrast(self, arrImages, fFactor):
		"""Adjust image contrast."""
		lstResult = []
		for img in arrImages:
			oPilImg = Image.fromarray(img if isinstance(img, np.ndarray) else np.array(img))
			oEnhancer = ImageEnhance.Contrast(oPilImg)
			oAdjusted = oEnhancer.enhance(fFactor)
			lstResult.append(np.array(oAdjusted))
		return np.array(lstResult)

	def _CalculateOverallScore(self, dctResults):
		"""
		Calculate overall robustness score.

		Args:
			dctResults: All test results.

		Returns:
			dict: Overall score information.
		"""
		fBaselineAcc = dctResults["baseline"].get("accuracy", 0)

		lstScores = []

		# Calculate average performance drop for each test
		for sKey in ["lighting", "noise", "blur", "contrast"]:
			lstTestResults = dctResults[sKey]
			fAvgAcc = np.mean([r["accuracy"] for r in lstTestResults])
			fScore = fAvgAcc / fBaselineAcc if fBaselineAcc > 0 else 0
			lstScores.append(fScore)

		fOverallScore = np.mean(lstScores)

		return {
			"score": float(fOverallScore),
			"baseline_accuracy": float(fBaselineAcc),
			"interpretation": self._InterpretScore(fOverallScore)
		}

	def _InterpretScore(self, fScore):
		"""
		Interpret robustness score.

		Args:
			fScore (float): Robustness score.

		Returns:
			str: Interpretation string.
		"""
		if fScore >= 0.9:
			return "Excellent - Model is highly robust"
		elif fScore >= 0.8:
			return "Good - Model shows good robustness"
		elif fScore >= 0.7:
			return "Fair - Model has moderate robustness"
		else:
			return "Poor - Model needs improvement in robustness"
