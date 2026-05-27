# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Report generator for emotion recognition evaluation.
Generates comprehensive evaluation reports in multiple formats.
"""

import os
from datetime import datetime


from evaluation.config import GetConfig
from evaluation.utils import GetTimestamp, SaveJson


class ReportGenerator:
	"""
	Generates comprehensive evaluation reports.
	Supports JSON, Markdown, and HTML formats.
	"""

	def __init__(self):
		"""Initialize report generator."""
		self._oConfig = GetConfig()
		self._sReportDir = self._oConfig.ReportDir
		self._lstEmotionLabels = self._oConfig.EmotionLabels
		self._dctData = {}

	def SetMetrics(self, dctMetrics):
		"""
		Set evaluation metrics data.

		Args:
			dctMetrics (dict): Metrics dictionary from MetricsCalculator.
		"""
		self._dctData["metrics"] = dctMetrics

	def SetSpeedResults(self, dctSpeed):
		"""
		Set speed benchmark results.

		Args:
			dctSpeed (dict): Speed results from SpeedBenchmark.
		"""
		self._dctData["speed"] = dctSpeed

	def SetRobustnessResults(self, dctRobustness):
		"""
		Set robustness analysis results.

		Args:
			dctRobustness (dict): Robustness results from RobustnessAnalyzer.
		"""
		self._dctData["robustness"] = dctRobustness

	def SetConfusionMatrixPath(self, sPath):
		"""
		Set confusion matrix image path.

		Args:
			sPath (str): Path to confusion matrix image.
		"""
		self._dctData["confusion_matrix_path"] = sPath

	def Generate(self, sFormat="markdown", sFileName=None):
		"""
		Generate evaluation report.

		Args:
			sFormat (str): Report format ("json", "markdown", "html").
			sFileName (str): Output file name without extension.

		Returns:
			str: Path to generated report.
		"""
		if sFileName is None:
			sFileName = f"evaluation_report_{GetTimestamp()}"

		if sFormat == "json":
			return self._GenerateJson(sFileName)
		elif sFormat == "markdown":
			return self._GenerateMarkdown(sFileName)
		elif sFormat == "html":
			return self._GenerateHtml(sFileName)
		else:
			raise ValueError(f"Unsupported format: {sFormat}")

	def _GenerateJson(self, sFileName):
		"""
		Generate JSON report.

		Args:
			sFileName (str): Output file name.

		Returns:
			str: Path to generated report.
		"""
		sPath = os.path.join(self._sReportDir, f"{sFileName}.json")

		sModelName = self._oConfig.Get("sModelName", "")
		dctReport = {
			"timestamp": datetime.now().isoformat(),
			"model_name": sModelName,
			"config": {
				"target_accuracy": self._oConfig.TargetAccuracy,
				"max_inference_time_ms": self._oConfig.MaxInferenceTime,
				"max_model_size_mb": self._oConfig.MaxModelSize
			},
			"data": self._dctData
		}

		SaveJson(dctReport, sPath)
		return sPath

	def _GenerateMarkdown(self, sFileName):
		"""
		Generate Markdown report.

		Args:
			sFileName (str): Output file name.

		Returns:
			str: Path to generated report.
		"""
		sPath = os.path.join(self._sReportDir, f"{sFileName}.md")

		sModelName = self._oConfig.Get("sModelName", "")
		sTitle = "表情识别模型评估报告"
		if sModelName:
			sTitle += f" ({sModelName})"

		lstLines = [
			f"# {sTitle}",
			"",
			f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
			"",
			"---",
			"",
			"## 1. 评估目标",
			"",
			f"- 目标准确率: ≥ {self._oConfig.TargetAccuracy * 100:.0f}%",
			f"- 最大推理时间: ≤ {self._oConfig.MaxInferenceTime:.0f}ms",
			f"- 最大模型大小: ≤ {self._oConfig.MaxModelSize:.0f}MB",
			""
		]

		# Metrics section
		if "metrics" in self._dctData:
			lstLines.extend(self._GenerateMetricsSection())

		# Speed section
		if "speed" in self._dctData:
			lstLines.extend(self._GenerateSpeedSection())

		# Robustness section
		if "robustness" in self._dctData:
			lstLines.extend(self._GenerateRobustnessSection())

		# Summary section
		lstLines.extend(self._GenerateSummarySection())

		# Write to file
		with open(sPath, "w", encoding="utf-8") as f:
			f.write("\n".join(lstLines))

		return sPath

	def _GenerateMetricsSection(self):
		"""Generate metrics section for markdown report."""
		dctMetrics = self._dctData["metrics"]

		lstLines = [
			"## 2. 性能指标",
			"",
			"### 2.1 整体性能",
			"",
			"| 指标 | 数值 |",
			"| --- | --- |",
			f"| 准确率 | {dctMetrics.get('accuracy', 0) * 100:.2f}% |",
			f"| 精确率 (Macro) | {dctMetrics.get('precision_macro', 0) * 100:.2f}% |",
			f"| 召回率 (Macro) | {dctMetrics.get('recall_macro', 0) * 100:.2f}% |",
			f"| F1分数 (Macro) | {dctMetrics.get('f1_macro', 0) * 100:.2f}% |",
			f"| 精确率 (Weighted) | {dctMetrics.get('precision_weighted', 0) * 100:.2f}% |",
			f"| 召回率 (Weighted) | {dctMetrics.get('recall_weighted', 0) * 100:.2f}% |",
			f"| F1分数 (Weighted) | {dctMetrics.get('f1_weighted', 0) * 100:.2f}% |",
			""
		]

		# Per-class metrics
		if "per_class" in dctMetrics:
			lstLines.extend([
				"### 2.2 各类别性能",
				"",
				"| 类别 | 精确率 | 召回率 | F1分数 | 支持数 |",
				"| --- | --- | --- | --- | --- |"
			])

			dctPerClass = dctMetrics["per_class"]
			for sLabel in self._lstEmotionLabels:
				if sLabel in dctPerClass:
					dctClassMetrics = dctPerClass[sLabel]
					lstLines.append(
						f"| {sLabel} | {dctClassMetrics['precision'] * 100:.2f}% | "
						f"{dctClassMetrics['recall'] * 100:.2f}% | "
						f"{dctClassMetrics['f1'] * 100:.2f}% | "
						f"{dctClassMetrics['support']} |"
					)

			lstLines.append("")

		# Target status
		bTargetMet = dctMetrics.get("target_met", False)
		sStatus = "✅ 达标" if bTargetMet else "❌ 未达标"
		lstLines.extend([
			"### 2.3 目标达成状态",
			"",
			f"**准确率目标**: {sStatus}",
			""
		])

		return lstLines

	def _GenerateSpeedSection(self):
		"""Generate speed section for markdown report."""
		dctSpeed = self._dctData["speed"]

		lstLines = [
			"## 3. 速度性能",
			""
		]

		# Single inference
		if "single_inference" in dctSpeed:
			dctSingle = dctSpeed["single_inference"]
			lstLines.extend([
				"### 3.1 单帧推理性能",
				"",
				"| 指标 | 数值 |",
				"| --- | --- |",
				f"| 平均推理时间 | {dctSingle['avg_time_ms']:.2f}ms |",
				f"| 最小推理时间 | {dctSingle['min_time_ms']:.2f}ms |",
				f"| 最大推理时间 | {dctSingle['max_time_ms']:.2f}ms |",
				f"| 标准差 | {dctSingle['std_time_ms']:.2f}ms |",
				f"| FPS | {dctSingle['fps']:.2f} |",
				""
			])

		# Model stats
		if "model_stats" in dctSpeed:
			dctStats = dctSpeed["model_stats"]
			lstLines.extend([
				"### 3.2 模型统计",
				"",
				"| 指标 | 数值 |",
				"| --- | --- |",
				f"| 设备 | {dctStats['device']} |",
				f"| 可训练参数 | {dctStats['trainable_params']:,} |",
				f"| 总参数 | {dctStats['total_params']:,} |",
				f"| 模型大小 | {dctStats['model_size_mb']:.2f}MB |",
				""
			])

		# Target status
		bTargetMet = dctSpeed.get("target_met", False)
		sStatus = "✅ 达标" if bTargetMet else "❌ 未达标"
		lstLines.extend([
			"### 3.3 目标达成状态",
			"",
			f"**推理速度目标**: {sStatus}",
			""
		])

		return lstLines

	def _GenerateRobustnessSection(self):
		"""Generate robustness section for markdown report."""
		dctRobustness = self._dctData["robustness"]

		lstLines = [
			"## 4. 鲁棒性分析",
			""
		]

		# Overall score
		if "overall_score" in dctRobustness:
			dctScore = dctRobustness["overall_score"]
			lstLines.extend([
				"### 4.1 鲁棒性评分",
				"",
				f"**综合评分**: {dctScore['score']:.2f}",
				"",
				f"**评价**: {dctScore['interpretation']}",
				""
			])

		# Detailed results
		lstTestTypes = [
			("lighting", "光照变化", "brightness_factor"),
			("noise", "噪声干扰", "noise_std"),
			("blur", "模糊处理", "blur_radius"),
			("contrast", "对比度变化", "contrast_factor")
		]

		for sKey, sTitle, sFactorKey in lstTestTypes:
			if sKey in dctRobustness:
				lstResults = dctRobustness[sKey]
				lstLines.extend([
					f"### 4.{lstTestTypes.index((sKey, sTitle, sFactorKey)) + 2} {sTitle}",
					"",
					f"| 因子 | 准确率 | F1分数 |",
					f"| --- | --- | --- |"
				])

				for dctResult in lstResults:
					lstLines.append(
						f"| {dctResult[sFactorKey]:.2f} | "
						f"{dctResult['accuracy'] * 100:.2f}% | "
						f"{dctResult['f1_score'] * 100:.2f}% |"
					)

				lstLines.append("")

		return lstLines

	def _GenerateSummarySection(self):
		"""Generate summary section for markdown report."""
		lstLines = [
			"## 5. 总结",
			""
		]

		# Collect status
		bAccTarget = False
		bSpeedTarget = False

		if "metrics" in self._dctData:
			bAccTarget = self._dctData["metrics"].get("target_met", False)

		if "speed" in self._dctData:
			bSpeedTarget = self._dctData["speed"].get("target_met", False)

		# Overall status
		if bAccTarget and bSpeedTarget:
			lstLines.append("**整体评估**: ✅ 模型满足所有设计目标")
		else:
			lstStatus = []
			if not bAccTarget:
				lstStatus.append("准确率未达标")
			if not bSpeedTarget:
				lstStatus.append("推理速度未达标")
			lstLines.append(f"**整体评估**: ❌ {', '.join(lstStatus)}")

		lstLines.append("")

		return lstLines

	def _GenerateHtml(self, sFileName):
		"""
		Generate HTML report.

		Args:
			sFileName (str): Output file name.

		Returns:
			str: Path to generated report.
		"""
		sPath = os.path.join(self._sReportDir, f"{sFileName}.html")

		# Generate markdown first and convert to HTML
		sMdPath = self._GenerateMarkdown(sFileName + "_temp")

		# Read markdown content
		with open(sMdPath, "r", encoding="utf-8") as f:
			sMdContent = f.read()

		# Convert to HTML (simple conversion)
		import markdown
		sHtmlContent = markdown.markdown(sMdContent, extensions=["tables"])

		# Create full HTML document
		sFullHtml = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>表情识别模型评估报告</title>
	<style>
		body {{
			font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
			max-width: 1000px;
			margin: 0 auto;
			padding: 20px;
			line-height: 1.6;
			color: #333;
		}}
		h1 {{
			color: #2c3e50;
			border-bottom: 3px solid #3498db;
			padding-bottom: 10px;
		}}
		h2 {{
			color: #34495e;
			border-bottom: 2px solid #bdc3c7;
			padding-bottom: 8px;
			margin-top: 30px;
		}}
		h3 {{
			color: #7f8c8d;
		}}
		table {{
			border-collapse: collapse;
			width: 100%;
			margin: 20px 0;
		}}
		th, td {{
			border: 1px solid #ddd;
			padding: 12px;
			text-align: left;
		}}
		th {{
			background-color: #3498db;
			color: white;
		}}
		tr:nth-child(even) {{
			background-color: #f2f2f2;
		}}
		hr {{
			border: 0;
			height: 1px;
			background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
			margin: 30px 0;
		}}
	</style>
</head>
<body>
	{sHtmlContent}
</body>
</html>"""

		# Write HTML
		with open(sPath, "w", encoding="utf-8") as f:
			f.write(sFullHtml)

		# Remove temporary markdown file
		os.remove(sMdPath)

		return sPath