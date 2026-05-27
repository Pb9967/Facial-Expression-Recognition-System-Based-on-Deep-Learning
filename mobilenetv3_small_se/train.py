# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Emotion recognition model training script.
Trains MobileNetV3 + SE Block based emotion recognition model.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sProjectRoot = Path(__file__).parent.parent
sys.path.append(str(sProjectRoot))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.amp import GradScaler, autocast

import matplotlib.pyplot as plt
from tqdm import tqdm

from mobilenetv3_small_se.config import ConfigLoader
from mobilenetv3_small_se.model import EmotionNet
from mobilenetv3_small_se.utils import ModelHelper


class TrainModel:
	"""
	Model training controller.
	Handles training loop, validation, and model saving.
	"""

	def __init__(self, dctConfig):
		"""
		Initialize training controller.

		Args:
			dctConfig (dict): Training configuration dictionary.
		"""
		self._dctConfig = dctConfig
		self._sDevice = dctConfig.get("sDevice", "cpu")
		self._logger = ModelHelper.SetupLogger()

	def GetDataLoaders(self):
		"""
		Create data loaders for training, validation, and testing.

		Returns:
			tuple: (train_loader, val_loader, test_loader)
		"""
		# Training data augmentation transforms
		trainTransform = transforms.Compose([
			transforms.Resize(self._dctConfig["tplInputSize"]),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(10),
			transforms.ColorJitter(
				brightness=0.2,
				contrast=0.2,
				saturation=0.2,
				hue=0.1
			),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225]
			)
		])

		# Validation and test transforms
		valTransform = transforms.Compose([
			transforms.Resize(self._dctConfig["tplInputSize"]),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225]
			)
		])

		# Load datasets
		trainDataset = ImageFolder(
			root=self._dctConfig["sTrainDir"],
			transform=trainTransform
		)
		valDataset = ImageFolder(
			root=self._dctConfig["sValDir"],
			transform=valTransform
		)
		testDataset = ImageFolder(
			root=self._dctConfig["sTestDir"],
			transform=valTransform
		)

		# Determine persistent_workers setting
		iNumWorkers = self._dctConfig.get("iNumWorkers", 4)
		bPersistentWorkers = iNumWorkers > 0

		# Create data loaders
		trainLoader = DataLoader(
			trainDataset,
			batch_size=self._dctConfig["iBatchSize"],
			shuffle=self._dctConfig.get("bShuffle", True),
			num_workers=iNumWorkers,
			pin_memory=self._dctConfig.get("bPinMemory", False),
			persistent_workers=bPersistentWorkers,
			prefetch_factor=self._dctConfig.get("iPrefetchFactor", 2)
		)

		valLoader = DataLoader(
			valDataset,
			batch_size=self._dctConfig["iBatchSize"],
			shuffle=False,
			num_workers=iNumWorkers,
			pin_memory=self._dctConfig.get("bPinMemory", False),
			persistent_workers=bPersistentWorkers,
			prefetch_factor=self._dctConfig.get("iPrefetchFactor", 2)
		)

		testLoader = DataLoader(
			testDataset,
			batch_size=self._dctConfig["iBatchSize"],
			shuffle=False,
			num_workers=iNumWorkers,
			pin_memory=self._dctConfig.get("bPinMemory", False),
			persistent_workers=bPersistentWorkers,
			prefetch_factor=self._dctConfig.get("iPrefetchFactor", 2)
		)

		return trainLoader, valLoader, testLoader

	def Run(self, model, trainLoader, valLoader):
		"""
		Execute training loop.

		Args:
			model (EmotionNet): Model to train.
			trainLoader (DataLoader): Training data loader.
			valLoader (DataLoader): Validation data loader.

		Returns:
			tuple: (trained_model, training_history)
		"""
		model.to(self._sDevice)

		# Setup loss function and optimizer
		criterion = nn.CrossEntropyLoss()
		optimizer = self._CreateOptimizer(model)
		scheduler = self._CreateScheduler(optimizer)

		# Setup AMP if enabled
		bUseAmp = (
			self._dctConfig.get("bUseAmp", False) and
			self._sDevice == "cuda"
		)
		scaler = GradScaler() if bUseAmp else None
		iGradAccumSteps = self._dctConfig.get("iGradientAccumSteps", 1)

		# Training history
		dctHistory = {
			"lstTrainLoss": [],
			"lstTrainAcc": [],
			"lstValLoss": [],
			"lstValAcc": []
		}

		self._logger.info(f"Starting training on device: {self._sDevice}")
		self._logger.info(f"Using AMP: {bUseAmp}")
		self._logger.info(f"Gradient accumulation steps: {iGradAccumSteps}")

		iEpochs = self._dctConfig["iEpochs"]

		for iEpoch in range(iEpochs):
			self._logger.info(f"Epoch [{iEpoch + 1}/{iEpochs}]")

			# Periodic GPU cache cleanup
			if (self._sDevice == "cuda" and
				(iEpoch + 1) % 10 == 0):
				torch.cuda.empty_cache()

			# Training phase
			fTrainLoss, fTrainAcc = self._TrainEpoch(
				model, trainLoader, criterion, optimizer,
				scaler, iGradAccumSteps, bUseAmp
			)

			# Validation phase
			fValLoss, fValAcc = self._ValidateEpoch(
				model, valLoader, criterion
			)

			# Update scheduler
			if scheduler is not None:
				scheduler.step()

			# Record history
			dctHistory["lstTrainLoss"].append(fTrainLoss)
			dctHistory["lstTrainAcc"].append(fTrainAcc)
			dctHistory["lstValLoss"].append(fValLoss)
			dctHistory["lstValAcc"].append(fValAcc)

			self._logger.info(f"Train Loss: {fTrainLoss:.4f} | Acc: {fTrainAcc:.2f}%")
			self._logger.info(f"Val Loss: {fValLoss:.4f} | Acc: {fValAcc:.2f}%")

			# Save best model
			self._SaveBestModel(model, dctHistory, iEpochs)

		# Save final model
		self._SaveFinalModel(model, iEpochs)

		return model, dctHistory

	def _TrainEpoch(self, model, trainLoader, criterion, optimizer,
					scaler, iGradAccumSteps, bUseAmp):
		"""
		Execute single training epoch.

		Returns:
			tuple: (avg_loss, accuracy)
		"""
		model.train()
		fTotalLoss = 0.0
		iCorrect = 0
		iTotal = 0

		optimizer.zero_grad()

		for iBatchIdx, (inputs, labels) in enumerate(
			tqdm(trainLoader, desc="Training")
		):
			inputs = inputs.to(self._sDevice)
			labels = labels.to(self._sDevice)

			if bUseAmp:
				fLoss, outputs = self._TrainStepAMP(
					model, inputs, labels, criterion,
					optimizer, scaler, iGradAccumSteps, iBatchIdx
				)
			else:
				fLoss, outputs = self._TrainStep(
					model, inputs, labels, criterion,
					optimizer, iGradAccumSteps, iBatchIdx
				)

			# Accumulate statistics (loss already scaled by iGradAccumSteps)
			fTotalLoss += fLoss * inputs.size(0) * iGradAccumSteps
			_, predicted = outputs.max(1)
			iTotal += labels.size(0)
			iCorrect += predicted.eq(labels).sum().item()

		# Handle remaining gradients at end of epoch
		iNumBatches = len(trainLoader)
		if iNumBatches % iGradAccumSteps != 0:
			if bUseAmp and scaler is not None:
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad()
			else:
				optimizer.step()
				optimizer.zero_grad()

		fAvgLoss = fTotalLoss / iTotal
		fAcc = 100.0 * iCorrect / iTotal

		return fAvgLoss, fAcc

	def _TrainStep(self, model, inputs, labels, criterion,
				   optimizer, iGradAccumSteps, iBatchIdx):
		"""
		Execute single training step without AMP.
		Implements gradient accumulation for effective batch size increase.
		"""
		outputs = model(inputs, bReturnLogits=True)
		fLoss = criterion(outputs, labels) / iGradAccumSteps
		fLoss.backward()

		# Update weights when gradient accumulation is complete
		if (iBatchIdx + 1) % iGradAccumSteps == 0:
			optimizer.step()
			optimizer.zero_grad()

		return fLoss.item(), outputs

	def _TrainStepAMP(self, model, inputs, labels, criterion,
					  optimizer, scaler, iGradAccumSteps, iBatchIdx):
		"""
		Execute single training step with AMP.
		Implements gradient accumulation with mixed precision training.
		"""
		with autocast(device_type='cuda'):
			outputs = model(inputs, bReturnLogits=True)
			fLoss = criterion(outputs, labels) / iGradAccumSteps

		scaler.scale(fLoss).backward()

		# Update weights when gradient accumulation is complete
		if (iBatchIdx + 1) % iGradAccumSteps == 0:
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad()

		return fLoss.item(), outputs

	def _ValidateEpoch(self, model, valLoader, criterion):
		"""
		Execute validation epoch.

		Returns:
			tuple: (avg_loss, accuracy)
		"""
		model.eval()
		fTotalLoss = 0.0
		iCorrect = 0
		iTotal = 0

		with torch.no_grad():
			for inputs, labels in tqdm(valLoader, desc="Validation"):
				inputs = inputs.to(self._sDevice)
				labels = labels.to(self._sDevice)

				outputs = model(inputs, bReturnLogits=True)
				fLoss = criterion(outputs, labels)

				fTotalLoss += fLoss.item() * inputs.size(0)
				_, predicted = outputs.max(1)
				iTotal += labels.size(0)
				iCorrect += predicted.eq(labels).sum().item()

		fAvgLoss = fTotalLoss / iTotal
		fAcc = 100.0 * iCorrect / iTotal

		return fAvgLoss, fAcc

	def _CreateOptimizer(self, model):
		"""Create optimizer based on configuration."""
		sOptimType = self._dctConfig.get("sOptimizer", "adam")
		fLr = self._dctConfig["fLearningRate"]
		fWeightDecay = self._dctConfig.get("fWeightDecay", 1e-4)

		if sOptimType == "adam":
			return optim.Adam(
				model.parameters(),
				lr=fLr,
				weight_decay=fWeightDecay
			)
		elif sOptimType == "sgd":
			return optim.SGD(
				model.parameters(),
				lr=fLr,
				momentum=0.9,
				weight_decay=fWeightDecay
			)
		else:
			raise ValueError(f"Unsupported optimizer: {sOptimType}")

	def _CreateScheduler(self, optimizer):
		"""Create learning rate scheduler."""
		sSchedulerType = self._dctConfig.get("sScheduler", "cosine")
		iEpochs = self._dctConfig["iEpochs"]

		if sSchedulerType == "cosine":
			return optim.lr_scheduler.CosineAnnealingLR(
				optimizer, T_max=iEpochs
			)
		elif sSchedulerType == "step":
			return optim.lr_scheduler.StepLR(
				optimizer, step_size=20, gamma=0.1
			)
		return None

	def _SaveBestModel(self, model, dctHistory, iEpochs):
		"""Save model if it achieves best validation accuracy."""
		lstValAcc = dctHistory["lstValAcc"]
		if len(lstValAcc) <= 1:
			return

		fCurrentAcc = lstValAcc[-1]
		fBestAcc = max(lstValAcc[:-1])

		if fCurrentAcc > fBestAcc:
			sPath = os.path.join(
				self._dctConfig["sWeightsDir"],
				f"{iEpochs}p_best_model.pth"
			)
			ModelHelper.SaveWeights(model, sPath)
			self._logger.info(f"Best model saved: {sPath}")

	def _SaveFinalModel(self, model, iEpochs):
		"""Save final model weights."""
		sPath = os.path.join(
			self._dctConfig["sWeightsDir"],
			"final_model.pth"
		)
		ModelHelper.SaveWeights(model, sPath)
		self._logger.info(f"Final model saved: {sPath}")


def main():
	"""Main entry point for training."""
	logger = ModelHelper.SetupLogger()
	logger.info("=== Emotion Recognition Model Training ===")

	# Load configuration
	configLoader = ConfigLoader()
	dctConfig = configLoader.GetConfig()

	# Update paths based on project structure
	sProjectRoot = Path(__file__).parent.parent
	dctConfig.update({
		"sDevice": "cuda" if torch.cuda.is_available() else "cpu",
		"sDataDir": str(sProjectRoot / "data_file" / "fer2013"),
		"sTrainDir": str(sProjectRoot / "data_file" / "fer2013" / "train"),
		"sValDir": str(sProjectRoot / "data_file" / "fer2013" / "val"),
		"sTestDir": str(sProjectRoot / "data_file" / "fer2013" / "test"),
		"iEpochs": 120,
		"iBatchSize": 64,
		"fLearningRate": 5e-4,
		"fWeightDecay": 1e-4,
		"sOptimizer": "adam",
		"sScheduler": "cosine",
		"iNumWorkers": 4,
		"bPinMemory": True,
		"bPersistentWorkers": True,
		"iPrefetchFactor": 2,
		"bShuffle": True,
		"bSaveBest": True,
		"bSaveFinal": True
	})

	iEpochs = dctConfig["iEpochs"]

	# Set output directories under train_Np/
	sModelDir = Path(__file__).parent
	dctConfig["sOutputDir"] = str(sModelDir / f"train_{iEpochs}p" / "outputs")
	dctConfig["sWeightsDir"] = str(sModelDir / f"train_{iEpochs}p" / "weights")
	dctConfig["sLogsDir"] = str(sModelDir / f"train_{iEpochs}p" / "logs")

	# Create all output directories in one place
	configLoader.UpdateConfig(dctConfig)
	configLoader.CreateOutputDirs()

	# Log configuration
	logger.info("Configuration:")
	for sKey, value in dctConfig.items():
		logger.info(f"  {sKey}: {value}")

	try:
		# Initialize trainer
		trainer = TrainModel(dctConfig)

		# Load data
		logger.info("Loading datasets...")
		trainLoader, valLoader, testLoader = trainer.GetDataLoaders()

		logger.info(f"Train samples: {len(trainLoader.dataset)}")
		logger.info(f"Val samples: {len(valLoader.dataset)}")
		logger.info(f"Test samples: {len(testLoader.dataset)}")
		logger.info(f"Classes: {EmotionNet().GetEmotionLabels()}")

		# Create model
		logger.info("Creating model...")
		model = EmotionNet(dctConfig)

		# Load pretrained weights if specified
		sPretrainPath = dctConfig.get("sPretrainedPath")
		if sPretrainPath and os.path.exists(sPretrainPath):
			logger.info(f"Loading pretrained: {sPretrainPath}")
			model = ModelHelper.LoadWeights(model, sPretrainPath)

		# Train model
		logger.info("Starting training...")
		model, dctHistory = trainer.Run(model, trainLoader, valLoader)

		# Evaluate model
		logger.info("Evaluating model...")
		EvaluateModel(model, testLoader, dctConfig)

		# Save training history
		sHistoryPath = os.path.join(
			dctConfig["sLogsDir"],
			f"train_history_{iEpochs}p.json"
		)
		with open(sHistoryPath, 'w', encoding='utf-8') as f:
			json.dump(dctHistory, f)
		logger.info(f"History saved: {sHistoryPath}")

		# Plot training curves
		PlotTrainingCurves(dctHistory, dctConfig["sLogsDir"], iEpochs)

		logger.info("=== Training Complete ===")

	except Exception as e:
		logger.error(f"Training error: {str(e)}")
		import traceback
		logger.error(traceback.format_exc())


def EvaluateModel(model, testLoader, dctConfig):
	"""
	Evaluate model on test set.

	Args:
		model (EmotionNet): Trained model.
		testLoader (DataLoader): Test data loader.
		dctConfig (dict): Configuration dictionary.

	Returns:
		float: Test accuracy.
	"""
	sDevice = dctConfig.get("sDevice", "cpu")
	model.to(sDevice)
	model.eval()

	iCorrect = 0
	iTotal = 0

	with torch.no_grad():
		for inputs, labels in tqdm(testLoader, desc="Testing"):
			inputs = inputs.to(sDevice)
			labels = labels.to(sDevice)

			outputs = model(inputs, bReturnLogits=True)
			_, predicted = outputs.max(1)

			iTotal += labels.size(0)
			iCorrect += predicted.eq(labels).sum().item()

	fAcc = 100.0 * iCorrect / iTotal
	logger = ModelHelper.SetupLogger()
	logger.info(f"Test Accuracy: {fAcc:.2f}%")

	return fAcc


def PlotTrainingCurves(dctHistory, sLogsDir, iEpochs):
	"""
	Plot and save training curves.

	Args:
		dctHistory (dict): Training history dictionary.
		sLogsDir (str): Directory to save plots.
		iEpochs (int): Number of training epochs.
	"""
	plt.figure(figsize=(12, 4))

	# Loss curve
	plt.subplot(1, 2, 1)
	plt.plot(dctHistory["lstTrainLoss"], label="Train Loss")
	plt.plot(dctHistory["lstValLoss"], label="Val Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Loss Curve")
	plt.legend()

	# Accuracy curve
	plt.subplot(1, 2, 2)
	plt.plot(dctHistory["lstTrainAcc"], label="Train Acc")
	plt.plot(dctHistory["lstValAcc"], label="Val Acc")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy (%)")
	plt.title("Accuracy Curve")
	plt.legend()

	plt.tight_layout()
	sPath = os.path.join(sLogsDir, f"training_curves_{iEpochs}p.png")
	plt.savefig(sPath)

	logger = ModelHelper.SetupLogger()
	logger.info(f"Training curves saved: training_curves_{iEpochs}p.png")


if __name__ == "__main__":
	main()