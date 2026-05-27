# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Speed benchmark for emotion recognition inference.
Measures FPS, latency, and throughput metrics.
"""

import time

import numpy as np
import torch

from evaluation.config import GetConfig
from evaluation.utils import CountParameters, Timer


class SpeedBenchmark:
	"""
	Benchmarks inference speed for emotion recognition model.
	Measures latency, FPS, and resource usage.
	"""

	def __init__(self, model=None, sDevice="cuda"):
		"""
		Initialize speed benchmark.

		Args:
			model (nn.Module): PyTorch model to benchmark.
			sDevice (str): Device to run benchmark on.
		"""
		self._oConfig = GetConfig()
		self._oModel = model
		self._sDevice = sDevice
		self._bUseAmp = torch.cuda.is_available() and sDevice == "cuda"

	def SetModel(self, model):
		"""
		Set model for benchmarking.

		Args:
			model (nn.Module): PyTorch model.
		"""
		self._oModel = model

	def Benchmark(self, tplInputShape=(1, 3, 224, 224)):
		"""
		Run comprehensive speed benchmark.

		Args:
			tplInputShape (tuple): Input tensor shape (B, C, H, W).

		Returns:
			dict: Benchmark results.
		"""
		if self._oModel is None:
			raise ValueError("Model not set. Call SetModel() first.")

		# Prepare model
		self._oModel.eval()
		self._oModel.to(self._sDevice)

		# Generate dummy input
		tensorInput = torch.randn(tplInputShape).to(self._sDevice)

		# Warmup
		self._Warmup(tensorInput)

		# Measure inference time
		dctResults = {}

		# Single inference benchmark
		dctResults["single_inference"] = self._BenchmarkSingle(tensorInput)

		# Batch inference benchmark
		dctResults["batch_inference"] = self._BenchmarkBatch(tplInputShape)

		# Model statistics
		dctResults["model_stats"] = self._GetModelStats()

		# Check if meets requirements
		dctResults["target_met"] = (
			dctResults["single_inference"]["avg_time_ms"] <= self._oConfig.MaxInferenceTime
		)

		return dctResults

	def _Warmup(self, tensorInput):
		"""
		Warmup model for accurate timing.

		Args:
			tensorInput: Dummy input tensor.
		"""
		iWarmupIters = self._oConfig.Get("iWarmupIterations")

		with torch.no_grad():
			for _ in range(iWarmupIters):
				if self._bUseAmp:
					with torch.amp.autocast('cuda'):
						_ = self._oModel(tensorInput)
				else:
					_ = self._oModel(tensorInput)

		# Synchronize if CUDA
		if self._sDevice == "cuda":
			torch.cuda.synchronize()

	def _BenchmarkSingle(self, tensorInput):
		"""
		Benchmark single inference latency.

		Args:
			tensorInput: Input tensor.

		Returns:
			dict: Single inference results.
		"""
		iIterations = self._oConfig.Get("iBenchmarkIterations")
		lstTimes = []

		with torch.no_grad():
			for _ in range(iIterations):
				oTimer = Timer().Start()

				if self._bUseAmp:
					with torch.amp.autocast('cuda'):
						_ = self._oModel(tensorInput)
				else:
					_ = self._oModel(tensorInput)

				if self._sDevice == "cuda":
					torch.cuda.synchronize()

				oTimer.Stop()
				lstTimes.append(oTimer.Elapsed())

		arrTimes = np.array(lstTimes)

		return {
			"iterations": iIterations,
			"avg_time_s": float(arrTimes.mean()),
			"avg_time_ms": float(arrTimes.mean() * 1000),
			"min_time_ms": float(arrTimes.min() * 1000),
			"max_time_ms": float(arrTimes.max() * 1000),
			"std_time_ms": float(arrTimes.std() * 1000),
			"fps": float(1.0 / arrTimes.mean())
		}

	def _BenchmarkBatch(self, tplInputShape):
		"""
		Benchmark batch inference performance.

		Args:
			tplInputShape (tuple): Base input shape.

		Returns:
			dict: Batch inference results.
		"""
		lstBatchSizes = [1, 2, 4, 8, 16]
		lstBatchResults = []

		iC, iH, iW = tplInputShape[1], tplInputShape[2], tplInputShape[3]

		for iBatchSize in lstBatchSizes:
			tensorInput = torch.randn(iBatchSize, iC, iH, iW).to(self._sDevice)

			# Warmup
			self._Warmup(tensorInput)

			# Benchmark
			lstTimes = []
			iIterations = max(10, self._oConfig.Get("iBenchmarkIterations") // iBatchSize)

			with torch.no_grad():
				for _ in range(iIterations):
					oTimer = Timer().Start()

					if self._bUseAmp:
						with torch.amp.autocast('cuda'):
							_ = self._oModel(tensorInput)
					else:
						_ = self._oModel(tensorInput)

					if self._sDevice == "cuda":
						torch.cuda.synchronize()

					oTimer.Stop()
					lstTimes.append(oTimer.Elapsed())

			fAvgTime = np.mean(lstTimes)
			fFps = iBatchSize / fAvgTime

			lstBatchResults.append({
				"batch_size": iBatchSize,
				"avg_time_ms": float(fAvgTime * 1000),
				"fps": float(fFps),
				"time_per_sample_ms": float(fAvgTime * 1000 / iBatchSize)
			})

		return lstBatchResults

	def _GetModelStats(self):
		"""
		Get model statistics.

		Returns:
			dict: Model statistics.
		"""
		dctStats = {
			"device": self._sDevice,
			"trainable_params": CountParameters(self._oModel),
			"total_params": sum(p.numel() for p in self._oModel.parameters())
		}

		# Model size estimation
		iParamSize = sum(p.numel() * p.element_size() for p in self._oModel.parameters())
		iBufferSize = sum(b.numel() * b.element_size() for b in self._oModel.buffers())
		dctStats["model_size_mb"] = (iParamSize + iBufferSize) / (1024 * 1024)

		# Check model size requirement
		dctStats["size_target_met"] = dctStats["model_size_mb"] <= self._oConfig.MaxModelSize

		return dctStats

	def ProfileMemory(self, tplInputShape=(1, 3, 224, 224)):
		"""
		Profile GPU memory usage (CUDA only).

		Args:
			tplInputShape (tuple): Input tensor shape.

		Returns:
			dict: Memory profiling results.
		"""
		if self._sDevice != "cuda":
			return {"error": "Memory profiling only available on CUDA device"}

		if self._oModel is None:
			raise ValueError("Model not set")

		# Reset memory stats
		torch.cuda.reset_peak_memory_stats()
		torch.cuda.empty_cache()

		# Prepare
		self._oModel.eval()
		tensorInput = torch.randn(tplInputShape).to(self._sDevice)

		# Measure memory before inference
		iMemBefore = torch.cuda.memory_allocated() / (1024 ** 2)  # MB

		# Run inference
		with torch.no_grad():
			if self._bUseAmp:
				with torch.cuda.amp.autocast():
					_ = self._oModel(tensorInput)
			else:
				_ = self._oModel(tensorInput)

		# Measure memory after inference
		torch.cuda.synchronize()
		iMemAfter = torch.cuda.memory_allocated() / (1024 ** 2)
		iMemPeak = torch.cuda.max_memory_allocated() / (1024 ** 2)

		return {
			"memory_before_mb": float(iMemBefore),
			"memory_after_mb": float(iMemAfter),
			"memory_peak_mb": float(iMemPeak),
			"memory_inference_mb": float(iMemPeak - iMemBefore)
		}
