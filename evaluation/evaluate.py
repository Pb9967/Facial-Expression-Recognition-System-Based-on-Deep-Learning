# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Evaluation entry script for emotion recognition model.
Provides comprehensive evaluation pipeline.
Supports dynamic model loading based on model path.
"""

import argparse
import importlib
import os
import re
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config import GetConfig
from evaluation.metrics import MetricsCalculator
from evaluation.confusion_matrix import ConfusionMatrixPlotter
from evaluation.speed import SpeedBenchmark
from evaluation.report import ReportGenerator


# Model name to Python module mapping
MODEL_MODULE_MAP = {
	"mobilenetv3_small_se": "mobilenetv3_small_se",
	"mobilenetv3_small_no_se": "mobilenetv3_small_no_se",
	"mobilenetv3_large_se": "mobilenetv3_large_se",
	"mobilenetv3_large_no_se": "mobilenetv3_large_no_se"
}


def ValidateAndParseModelPath(sModelPath):
	"""
	Validate model path format and extract model name.

	Expected relative path format containing pattern:
		{model_name}/train_{N}p/weights/final_model.pth

	Args:
		sModelPath (str): Path to model weights as provided by user.

	Returns:
		tuple: (model_name, error_message).
			   If validation fails, model_name is None and error_message
			   contains the warning text printed to console.
	"""
	# Reject absolute paths
	if os.path.isabs(sModelPath):
		return (
			None,
			"路径格式错误：检测到绝对路径。请输入相对路径 "
			"（如 ../mobilenetv3_small_se/train_120p/weights/final_model.pth）"
		)

	# Normalize path for consistent checking
	sNormalized = os.path.normpath(sModelPath).replace("\\", "/")

	# Validate path contains the expected pattern:
	#   {model_name}/train_{digits}p/weights/final_model.pth
	sPattern = (
		r"(mobilenetv3_small_se|mobilenetv3_small_no_se|"
		r"mobilenetv3_large_se|mobilenetv3_large_no_se)"
		r"/train_\d+p/weights/final_model\.pth"
	)
	match = re.search(sPattern, sNormalized)

	if not match:
		return (
			None,
			"路径格式错误：无法识别模型路径格式。\n"
			"正确格式示例: ../mobilenetv3_small_se/train_120p/weights/final_model.pth\n"
			"或: ./mobilenetv3_large_se/train_50p/weights/final_model.pth"
		)

	sModelName = match.group(1)
	return sModelName, None


def LoadModel(sModelPath, sDevice, sModelName):
	"""
	Load trained model from checkpoint dynamically based on model name.

	Args:
		sModelPath (str): Absolute path to model weights.
		sDevice (str): Device to load model on.
		sModelName (str): Model name for dynamic module import.

	Returns:
		nn.Module: Loaded model.

	Raises:
		ValueError: If model name is unknown.
		RuntimeError: If model loading fails.
	"""
	sModuleName = MODEL_MODULE_MAP.get(sModelName)
	if not sModuleName:
		raise ValueError(f"Unknown model name: {sModelName}")

	# Dynamic import of the correct model module
	try:
		modelModule = importlib.import_module(f"{sModuleName}.model")
		CreateModel = modelModule.CreateModel
	except Exception as e:
		raise RuntimeError(
			f"Failed to import model module '{sModuleName}.model': {str(e)}"
		)

	# Create model instance (backbone type is handled by each model's defaults)
	dctConfig = {"fDropout": 0.2}
	oModel = CreateModel(dctConfig, bPretrained=False)

	# Load weights
	try:
		oModel.load_state_dict(torch.load(sModelPath, map_location=sDevice))
	except Exception as e:
		raise RuntimeError(f"Failed to load weights from {sModelPath}: {str(e)}")

	oModel.to(sDevice)
	oModel.eval()

	return oModel


def PrepareDataLoader(sDataDir, iBatchSize=32):
	"""
	Prepare test data loader.

	Args:
		sDataDir (str): Path to test data directory.
		iBatchSize (int): Batch size.

	Returns:
		DataLoader: Test data loader.
		list: Class names.
	"""
	oTransform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	oDataset = datasets.ImageFolder(sDataDir, transform=oTransform)
	oLoader = DataLoader(oDataset, batch_size=iBatchSize, shuffle=False, num_workers=4)

	return oLoader, oDataset.classes


def EvaluateMetrics(model, oLoader, sDevice):
	"""
	Evaluate model metrics.

	Args:
		model (nn.Module): Model to evaluate.
		oLoader (DataLoader): Test data loader.
		sDevice (str): Device.

	Returns:
		dict: Metrics results.
	"""
	oMetrics = MetricsCalculator()
	model.eval()

	with torch.no_grad():
		for images, labels in oLoader:
			images = images.to(sDevice)
			labels = labels.to(sDevice)

			outputs = model(images)
			predictions = torch.argmax(outputs, dim=1)

			oMetrics.AddBatch(predictions.cpu(), labels.cpu())

	return oMetrics.Compute()


def RunEvaluation(sModelPath, sDataDir, sOutputDir=None, sModelName=None):
	"""
	Run complete evaluation pipeline.

	Args:
		sModelPath (str): Absolute path to model weights.
		sDataDir (str): Path to test data.
		sOutputDir (str): Optional override output directory for reports.
		sModelName (str): Model name (e.g., 'mobilenetv3_small_se').

	Returns:
		dict: Complete evaluation results.
	"""
	# Configuration (paths already updated in main() via SetModelName)
	oConfig = GetConfig()

	if sOutputDir:
		os.makedirs(sOutputDir, exist_ok=True)

	# Device
	sDevice = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {sDevice}")

	# Load model
	print(f"\n[1/5] Loading model: {sModelName} ...")
	model = LoadModel(sModelPath, sDevice, sModelName)
	print("Model loaded successfully.")

	# Prepare data
	print("\n[2/5] Preparing test data...")
	oLoader, lstClasses = PrepareDataLoader(sDataDir)
	print(f"Test samples: {len(oLoader.dataset)}, Classes: {len(lstClasses)}")

	# Initialize report generator
	oReport = ReportGenerator()

	# Evaluate metrics
	print("\n[3/5] Evaluating metrics...")
	dctMetrics = EvaluateMetrics(model, oLoader, sDevice)
	oReport.SetMetrics(dctMetrics)
	print(f"Accuracy: {dctMetrics['accuracy'] * 100:.2f}%")
	print(f"F1-Score: {dctMetrics['f1_macro'] * 100:.2f}%")

	# Confusion matrix
	print("\n[4/5] Generating confusion matrix...")
	oMetricsCalc = MetricsCalculator()
	oMetricsCalc._lstYPred = []
	oMetricsCalc._lstYTrue = []

	# Re-collect predictions for confusion matrix
	model.eval()
	with torch.no_grad():
		for images, labels in oLoader:
			images = images.to(sDevice)
			outputs = model(images)
			predictions = torch.argmax(outputs, dim=1)
			oMetricsCalc.AddBatch(predictions.cpu(), labels)

	arrConfMatrix = oMetricsCalc.GetConfusionMatrix()
	oPlotter = ConfusionMatrixPlotter()
	sCmPath = oPlotter.PlotBoth(arrConfMatrix, sTitle=f"Emotion Recognition ({sModelName})")
	oReport.SetConfusionMatrixPath(sCmPath)
	print(f"Confusion matrix saved to: {sCmPath}")

	# Speed benchmark
	print("\n[5/5] Running speed benchmark...")
	oBenchmarker = SpeedBenchmark(model, sDevice)
	dctSpeed = oBenchmarker.Benchmark()
	oReport.SetSpeedResults(dctSpeed)
	print(f"Average inference time: {dctSpeed['single_inference']['avg_time_ms']:.2f}ms")
	print(f"FPS: {dctSpeed['single_inference']['fps']:.2f}")

	# Generate report
	print("\nGenerating evaluation report...")
	sReportPath = oReport.Generate(sFormat="markdown")
	print(f"Report saved to: {sReportPath}")

	# Summary
	print("\n" + "=" * 50)
	print("EVALUATION SUMMARY")
	print("=" * 50)
	print(f"Model: {sModelName}")
	print(f"Accuracy: {dctMetrics['accuracy'] * 100:.2f}% (Target: {oConfig.TargetAccuracy * 100:.0f}%)")
	print(f"Inference Time: {dctSpeed['single_inference']['avg_time_ms']:.2f}ms (Target: ≤{oConfig.MaxInferenceTime:.0f}ms)")
	print(f"Model Size: {dctSpeed['model_stats']['model_size_mb']:.2f}MB (Target: ≤{oConfig.MaxModelSize:.0f}MB)")
	print("=" * 50)

	bAccTarget = dctMetrics['accuracy'] >= oConfig.TargetAccuracy
	bSpeedTarget = dctSpeed['single_inference']['avg_time_ms'] <= oConfig.MaxInferenceTime
	bSizeTarget = dctSpeed['model_stats']['model_size_mb'] <= oConfig.MaxModelSize

	if bAccTarget and bSpeedTarget and bSizeTarget:
		print("✅ All targets met!")
	else:
		print("❌ Some targets not met")
		if not bAccTarget:
			print("  - Accuracy target not met")
		if not bSpeedTarget:
			print("  - Inference time target not met")
		if not bSizeTarget:
			print("  - Model size target not met")

	return {
		"metrics": dctMetrics,
		"speed": dctSpeed,
		"report_path": sReportPath
	}


def main():
	"""Main entry point."""
	# 调试模式：如果没有命令行参数，使用内部默认路径
	if len(sys.argv) == 1:
		print("Debug mode: using default paths")
		# 请根据实际路径修改以下变量
		sModelPath = "../mobilenetv3_small_se/train_100p/weights/final_model.pth"
		sDataDir = "../data_file/fer2013/test"
		sOutputDir = None  # 使用 config 中模型相关的输出目录

		# 1) 验证模型路径格式
		sModelName, sError = ValidateAndParseModelPath(sModelPath)
		if sError:
			print(f"[警告] {sError}")
			sys.exit(1)

		print(f"检测到模型: {sModelName}")

		# 2) 更新配置：在 evaluation/outputs/ 和 evaluation/reports/ 下创建模型专属子目录
		oConfig = GetConfig()
		oConfig.SetModelName(sModelName)

		# 3) 将相对路径转为绝对路径供后续加载使用
		sScriptDir = os.path.dirname(os.path.abspath(__file__))
		sAbsModelPath = os.path.normpath(os.path.join(sScriptDir, sModelPath))
		sAbsDataDir = os.path.normpath(os.path.join(sScriptDir, sDataDir))

		RunEvaluation(sAbsModelPath, sAbsDataDir, sOutputDir, sModelName)

	else:
		oParser = argparse.ArgumentParser(description="Evaluate emotion recognition model")
		oParser.add_argument(
			"--model", type=str, required=True,
			help="Path to model weights (relative path like ../mobilenetv3_small_se/train_120p/weights/final_model.pth)"
		)
		oParser.add_argument("--data", type=str, required=True, help="Path to test data directory")
		oParser.add_argument(
			"--output", type=str, default=None,
			help="Output directory for reports (optional, overrides default)"
		)
		oArgs = oParser.parse_args()

		# 1) 验证模型路径格式
		sModelName, sError = ValidateAndParseModelPath(oArgs.model)
		if sError:
			print(f"[警告] {sError}")
			sys.exit(1)

		print(f"检测到模型: {sModelName}")

		# 2) 更新配置：创建模型专属输出子目录
		oConfig = GetConfig()
		oConfig.SetModelName(sModelName)

		# 3) 将相对路径转为绝对路径
		sScriptDir = os.path.dirname(os.path.abspath(__file__))
		sAbsModelPath = os.path.normpath(os.path.join(sScriptDir, oArgs.model))
		sAbsDataDir = os.path.normpath(os.path.join(sScriptDir, oArgs.data))

		RunEvaluation(sAbsModelPath, sAbsDataDir, oArgs.output, sModelName)


if __name__ == "__main__":
	main()