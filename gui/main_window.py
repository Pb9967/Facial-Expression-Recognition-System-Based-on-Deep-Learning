# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Main window module.
Optimized for real-time video processing with async frame processing.
Supports dynamic model switching across multiple MobileNetV3 variants.
"""
import sys
import os
import cv2
import torch
import dlib
from collections import deque
from datetime import datetime
from pathlib import Path
import logging
import threading
import time
import queue
import importlib

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import (
	QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
	QPushButton, QFileDialog, QSplitter,
	QListWidget, QListWidgetItem, QMessageBox, QGroupBox
)

from widgets.image_widget import ImageWidget
from widgets.camera_widget import CameraThread
from gui_utils import SafeDrawEmotionResult

# Add project root to path
sProjectRoot = Path(__file__).parent.parent
sys.path.append(str(sProjectRoot))

from face_detection_alignment.detector import FaceDetector
from face_detection_alignment.aligner import FaceAligner
from mobilenetv3_small_se.utils import ModelHelper

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


TRAINS_EPOCH = 100


# Model registry: mapping display name to module and config
MODEL_REGISTRY = {
	"mobilenetv3_small_se": {
		"module": "mobilenetv3_small_se",
		"config": {"sBackbone": "small"}
	},
	"mobilenetv3_small_no_se": {
		"module": "mobilenetv3_small_no_se",
		"config": {"sBackbone": "small"}
	},
	"mobilenetv3_large_se": {
		"module": "mobilenetv3_large_se",
		"config": {"sBackbone": "large"}
	},
	"mobilenetv3_large_no_se": {
		"module": "mobilenetv3_large_no_se",
		"config": {"sBackbone": "large"}
	}
}


# Configure matplotlib font for cross-platform compatibility
plt.rcParams['font.sans-serif'] = [
	'SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei',
	'DejaVu Sans', 'Noto Sans CJK SC', 'Arial Unicode MS'
]
plt.rcParams['axes.unicode_minus'] = False

# Configure unified logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	filename='gui_errors.log',
	filemode='w'
)
logger = logging.getLogger(__name__)


class FrameProcessorThread(QThread):
	"""
	Background thread for frame processing.
	Processes frames from queue and emits results to main thread.
	Supports dynamic model replacement for multi-model switching.
	"""

	# Signal: processed frame, emotion, confidence
	result_ready = pyqtSignal(object, str, float)
	error_occurred = pyqtSignal(str)

	def __init__(self, detector, aligner, model, sDevice, lstEmotionLabels):
		"""
		Initialize processor thread.

		Args:
			detector: Face detector instance.
			aligner: Face aligner instance.
			model: Emotion recognition model (can be None initially).
			sDevice: Device string.
			lstEmotionLabels: List of emotion labels (can be None initially).
		"""
		super().__init__()
		self._detector = detector
		self._aligner = aligner
		self._model = model
		self._sDevice = sDevice
		self._lstEmotionLabels = lstEmotionLabels
		self._frameQueue = queue.Queue(maxsize=2)  # Limit queue size
		self._bRunning = False

	def setModel(self, model, lstEmotionLabels):
		"""
		Update model and emotion labels dynamically at runtime.

		Args:
			model: New emotion recognition model instance.
			lstEmotionLabels: Emotion labels corresponding to the new model.
		"""
		self._model = model
		self._lstEmotionLabels = lstEmotionLabels

	def addFrame(self, frame):
		"""
		Add frame to processing queue.
		Non-blocking, drops frame if queue is full.

		Args:
			frame: Input frame.
		"""
		if self._bRunning:
			try:
				# Non-blocking put, drop frame if queue full
				self._frameQueue.put_nowait(frame)
			except queue.Full:
				pass  # Drop frame to maintain real-time

	def run(self):
		"""Thread main loop for frame processing."""
		self._bRunning = True

		while self._bRunning:
			try:
				# Get frame from queue (blocking with timeout)
				frame = self._frameQueue.get(timeout=0.1)

				if frame is None or frame.size == 0:
					continue

				# Process frame
				displayFrame, sEmotion, fConf = self._ProcessFrame(frame)

				# Emit result
				self.result_ready.emit(displayFrame, sEmotion, fConf)

			except queue.Empty:
				continue
			except Exception as e:
				logger.error(f"Frame processing error: {str(e)}")

	def stop(self):
		"""Stop thread."""
		self._bRunning = False
		self.wait()

	def _ProcessFrame(self, frame):
		"""
		Process single frame for emotion recognition.

		Args:
			frame: Input frame.

		Returns:
			tuple: (displayFrame, emotion, confidence)
		"""
		# If model not loaded yet, return original frame without processing
		if self._model is None or self._lstEmotionLabels is None:
			return frame.copy(), "unknown", 0.0

		displayFrame = frame.copy()
		sEmotion = "unknown"
		fConf = 0.0

		try:
			# Face detection
			lstFaces = self._detector.detect_faces(frame)

			for (iX1, iY1, iX2, iY2, fDetConf) in lstFaces:
				# Clamp coordinates
				iH, iW = frame.shape[:2]
				iX1 = max(0, min(int(iX1), iW - 1))
				iY1 = max(0, min(int(iY1), iH - 1))
				iX2 = max(0, min(int(iX2), iW - 1))
				iY2 = max(0, min(int(iY2), iH - 1))

				if iX2 <= iX1 or iY2 <= iY1:
					continue

				# Face alignment
				rect = dlib.rectangle(iX1, iY1, iX2, iY2)
				lstLandmarks = self._aligner.detect_landmarks(frame, [rect])

				if not lstLandmarks:
					continue

				alignedFace = self._aligner.align_face(frame, lstLandmarks[0])
				if alignedFace is None:
					continue

				# Emotion prediction
				tensor = ModelHelper.PreprocessImage(alignedFace)
				tensor = tensor.to(self._sDevice)

				with torch.no_grad():
					output = self._model(tensor)

				# Postprocess predictions
				sEmotion, fConf = ModelHelper.PostprocessPredictions(
					output[0], self._lstEmotionLabels
				)

				# Draw result
				displayFrame = SafeDrawEmotionResult(
					displayFrame, sEmotion, fConf, (iX1, iY1, iX2, iY2)
				)

				break  # Only process first face

		except Exception as e:
			logger.warning(f"Frame processing failed: {str(e)}")

		return displayFrame, sEmotion, fConf


class EmotionRecognitionGUI(QMainWindow):
	"""
	Main window for emotion recognition system.
	Optimized for real-time video processing with async processing.
	Supports switching between multiple MobileNetV3 model variants.
	"""

	# Emotion labels (class constant)
	CLST_EMOTION_LABELS = [
		"angry", "disgust", "fear", "happy",
		"sad", "surprise", "neutral"
	]

	def __init__(self):
		"""Initialize main window."""
		super().__init__()
		self.setWindowTitle("基于深度学习的人脸情绪识别系统")
		self.setGeometry(100, 100, 1200, 800)
		self.setMinimumSize(800, 600)

		# Backend components
		self._detector = None
		self._aligner = None
		self._model = None
		self._sDevice = "cuda" if torch.cuda.is_available() else "cpu"

		# UI components
		self._imageDisplay = None
		self._btnCamera = None
		self._btnImage = None
		self._btnVideo = None
		self._btnStop = None
		self._recordList = None
		self._statusBar = None
		self._dctModelButtons = {}  # Model selection buttons registry

		# State management
		self._sCurrentMode = "camera"
		self._cameraThread = None
		self._videoThread = None
		self._processorThread = None
		self._lstRecords = []
		self._deqEmotionHistory = deque(maxlen=100)
		self._bDataChanged = False
		self._processLock = threading.Lock()

		# Model selection state
		self._bModelSelected = False
		self._sSelectedModelName = None
		self._bIsProcessing = False

		# Performance optimization
		self._iRecordInterval = 5  # Add record every 5 frames
		self._iRecordCount = 0

		# Pending record batch for UI optimization
		self._lstPendingRecords = []
		self._recordTimer = QTimer()
		self._recordTimer.timeout.connect(self._FlushRecords)

		# FPS tracking
		self._fLastFrameTime = time.time()
		self._deqFrameTimes = deque(maxlen=30)
		self._iDisplayFrameCount = 0

		# Initialize components
		self._InitBackend()
		self._InitUI()

		# Setup curve update timer
		self._curveTimer = QTimer()
		self._curveTimer.timeout.connect(self._UpdateCurve)
		self._curveTimer.start(500)

	def _InitBackend(self):
		"""Initialize backend components (detector, aligner, processor thread)."""
		logger.info("Initializing backend components")

		# Initialize face detector and aligner
		try:
			self._detector = FaceDetector()
			self._aligner = FaceAligner()
			logger.info("Face detector and aligner initialized")
		except Exception as e:
			logger.error(f"Failed to initialize face detection: {str(e)}")
			QMessageBox.critical(
				self, "错误",
				f"初始化人脸检测/对齐模块失败:\n{str(e)}"
			)
			sys.exit(1)

		# Initialize processor thread (model will be set after user selection)
		self._processorThread = FrameProcessorThread(
			self._detector, self._aligner,
			None, self._sDevice, None
		)
		self._processorThread.result_ready.connect(self._OnResultReady)
		self._processorThread.error_occurred.connect(self._OnError)

	def _InitUI(self):
		"""Build user interface."""
		centralWidget = QWidget()
		self.setCentralWidget(centralWidget)
		mainLayout = QHBoxLayout(centralWidget)

		# Left panel: image display and buttons
		leftWidget = QWidget()
		leftLayout = QVBoxLayout(leftWidget)
		self._imageDisplay = ImageWidget()
		leftLayout.addWidget(self._imageDisplay)

		# Model selection group
		modelGroup = QGroupBox("模型选择 (必须先选择模型才能进行识别)")
		modelLayout = QHBoxLayout()
		for sModelName in MODEL_REGISTRY.keys():
			btn = QPushButton(sModelName)
			btn.clicked.connect(
				lambda checked, name=sModelName: self._SelectModel(name)
			)
			modelLayout.addWidget(btn)
			self._dctModelButtons[sModelName] = btn
		modelGroup.setLayout(modelLayout)
		leftLayout.addWidget(modelGroup)

		# Button layout
		btnLayout = QHBoxLayout()
		self._btnCamera = QPushButton("打开摄像头")
		self._btnCamera.clicked.connect(self._OpenCamera)
		self._btnImage = QPushButton("选择图片")
		self._btnImage.clicked.connect(self._OpenImage)
		self._btnVideo = QPushButton("选择视频")
		self._btnVideo.clicked.connect(self._OpenVideo)
		self._btnStop = QPushButton("停止")
		self._btnStop.clicked.connect(self._StopProcessing)

		btnLayout.addWidget(self._btnCamera)
		btnLayout.addWidget(self._btnImage)
		btnLayout.addWidget(self._btnVideo)
		btnLayout.addWidget(self._btnStop)
		leftLayout.addLayout(btnLayout)

		# Right panel: curve and records
		rightWidget = QWidget()
		rightLayout = QVBoxLayout(rightWidget)

		# Emotion curve group
		curveGroup = QGroupBox("情绪变化曲线")
		curveLayout = QVBoxLayout()
		self._CreateCurveCanvas()
		curveLayout.addWidget(self._figureCanvas)
		curveGroup.setLayout(curveLayout)
		rightLayout.addWidget(curveGroup)

		# Recognition record group
		recordGroup = QGroupBox("识别记录")
		recordLayout = QVBoxLayout()
		self._recordList = QListWidget()
		recordLayout.addWidget(self._recordList)
		recordGroup.setLayout(recordLayout)
		rightLayout.addWidget(recordGroup)

		# Splitter
		splitter = QSplitter(Qt.Horizontal)
		splitter.addWidget(leftWidget)
		splitter.addWidget(rightWidget)
		splitter.setSizes([800, 400])
		mainLayout.addWidget(splitter)

		# Menu bar
		menubar = self.menuBar()
		fileMenu = menubar.addMenu("文件")
		exitAction = QAction("退出", self)
		exitAction.triggered.connect(self.close)
		fileMenu.addAction(exitAction)

		helpMenu = menubar.addMenu("帮助")
		aboutAction = QAction("关于", self)
		aboutAction.triggered.connect(self._ShowAbout)
		helpMenu.addAction(aboutAction)

		# Status bar
		self._statusBar = self.statusBar()
		self._statusBar.showMessage("就绪 - 请先选择一个模型")

	def _SelectModel(self, sModelName):
		"""
		Select and load emotion recognition model dynamically.

		Uses the global TRAINS_EPOCH variable to construct the weight path:
			{model_dir}/train_{TRAINS_EPOCH}p/weights/final_model.pth

		Args:
			sModelName: Name of the model to load (key in MODEL_REGISTRY).
		"""
		# Prevent model switching during active prediction
		if self._bIsProcessing:
			print(f"[警告] 当前正在预测中，无法切换模型。请先点击停止按钮。")
			return

		logger.info(f"Selecting model: {sModelName}")

		dctModelInfo = MODEL_REGISTRY.get(sModelName)
		if not dctModelInfo:
			print(f"[错误] 未知模型: {sModelName}")
			return

		sModuleName = dctModelInfo["module"]

		# Dynamic import of model module
		try:
			modelModule = importlib.import_module(f"{sModuleName}.model")
			utilsModule = importlib.import_module(f"{sModuleName}.utils")
			EmotionNet = modelModule.EmotionNet
			ModelHelperCls = utilsModule.ModelHelper
		except Exception as e:
			logger.error(f"Failed to import model module {sModuleName}: {str(e)}")
			QMessageBox.critical(
				self, "错误",
				f"导入模型模块失败:\n{str(e)}"
			)
			return

		# Create model instance and load weights
		try:
			dctConfig = dctModelInfo["config"]
			model = EmotionNet(dctConfig)

			# Build weight path using global TRAINS_EPOCH
			sWeightPath = (
				sProjectRoot / sModuleName /
				f"train_{TRAINS_EPOCH}p" / "weights" / "final_model.pth"
			)

			if not sWeightPath.exists():
				raise FileNotFoundError(f"Weight file not found: {sWeightPath}")

			model = ModelHelperCls.LoadWeights(model, str(sWeightPath))
			model.to(self._sDevice)
			model.eval()

		except Exception as e:
			logger.error(f"Failed to load model {sModelName}: {str(e)}")
			QMessageBox.critical(
				self, "错误",
				f"加载模型 [{sModelName}] 失败:\n{str(e)}"
			)
			self._bModelSelected = False
			self._sSelectedModelName = None
			self._statusBar.showMessage(f"模型加载失败: {sModelName}")
			return

		# Update model reference and processor thread
		self._model = model
		self._sSelectedModelName = sModelName
		self._bModelSelected = True

		# Update processor thread with new model
		if self._processorThread is not None:
			self._processorThread.setModel(model, model.GetEmotionLabels())

		# Update UI feedback
		self._UpdateModelButtonStyles()
		self._statusBar.showMessage(
			f"已选择模型: {sModelName} | Epoch={TRAINS_EPOCH}"
		)
		print(f"[信息] 模型切换成功: {sModelName} | 权重路径: {sWeightPath}")
		logger.info(f"Model loaded successfully: {sModelName}")

	def _UpdateModelButtonStyles(self):
		"""Highlight the currently selected model button."""
		for sName, btn in self._dctModelButtons.items():
			if sName == self._sSelectedModelName:
				btn.setStyleSheet(
					"QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
				)
			else:
				btn.setStyleSheet("")

	def _CreateCurveCanvas(self):
		"""Create matplotlib canvas for emotion curve."""
		self._curveFig = Figure(figsize=(5, 2), dpi=100)
		self._curveAx = self._curveFig.add_subplot(111)
		self._curveAx.set_title("情绪变化趋势")
		self._curveAx.set_xlabel("帧序号")
		self._curveAx.set_ylabel("情绪类别")
		self._curveAx.set_ylim(0, len(self.CLST_EMOTION_LABELS) - 1)
		self._curveAx.set_yticks(range(len(self.CLST_EMOTION_LABELS)))
		self._curveAx.set_yticklabels(self.CLST_EMOTION_LABELS, fontsize=8)
		self._figureCanvas = FigureCanvasQTAgg(self._curveFig)

	def _UpdateCurve(self):
		"""Update emotion curve (data-driven)."""
		if not self._deqEmotionHistory or not self._bDataChanged:
			return

		try:
			lstData = list(self._deqEmotionHistory)
			self._curveAx.clear()
			self._curveAx.plot(lstData, 'b-', linewidth=2)
			self._curveAx.set_title("情绪变化趋势")
			self._curveAx.set_xlabel("帧序号")
			self._curveAx.set_ylabel("情绪类别")
			self._curveAx.set_ylim(0, len(self.CLST_EMOTION_LABELS) - 1)
			self._curveAx.set_yticks(range(len(self.CLST_EMOTION_LABELS)))
			self._curveAx.set_yticklabels(self.CLST_EMOTION_LABELS, fontsize=8)
			self._figureCanvas.draw()
			self._bDataChanged = False
		except Exception as e:
			logger.error(f"Failed to update curve: {str(e)}")

	def _AddRecord(self, sEmotion, fConfidence, sTimestamp=None):
		"""
		Add recognition record to pending batch.

		Args:
			sEmotion: Emotion label.
			fConfidence: Confidence score.
			sTimestamp: Optional timestamp string.
		"""
		if sTimestamp is None:
			sTimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

		self._lstPendingRecords.append((sTimestamp, sEmotion, fConfidence))

		if not self._recordTimer.isActive():
			self._recordTimer.start(500)

	def _FlushRecords(self):
		"""Flush pending records to list widget in batch."""
		if not self._lstPendingRecords:
			self._recordTimer.stop()
			return

		for sTimestamp, sEmotion, fConf in self._lstPendingRecords:
			sText = f"{sTimestamp} - {sEmotion} ({fConf:.1f}%)"
			item = QListWidgetItem(sText)
			self._recordList.addItem(item)
			self._lstRecords.append((sTimestamp, sEmotion, fConf))

		self._recordList.scrollToBottom()
		self._lstPendingRecords.clear()

	def _OnResultReady(self, displayFrame, sEmotion, fConfidence):
		"""
		Handle processed result from background thread.

		Args:
			displayFrame: Processed frame with annotations.
			sEmotion: Detected emotion.
			fConfidence: Confidence score.
		"""
		# Update display
		if displayFrame is not None:
			self._imageDisplay.set_image(displayFrame)

		# Update emotion history
		if sEmotion != "unknown":
			try:
				iIdx = self.CLST_EMOTION_LABELS.index(sEmotion)
			except ValueError:
				iIdx = 6
			self._deqEmotionHistory.append(iIdx)
			self._bDataChanged = True

		# Add record with sampling
		self._iRecordCount += 1
		if self._iRecordCount % self._iRecordInterval == 0:
			self._AddRecord(sEmotion, fConfidence)

		# Calculate and display FPS
		fCurrentTime = time.time()
		fFrameTime = fCurrentTime - self._fLastFrameTime
		self._fLastFrameTime = fCurrentTime
		self._deqFrameTimes.append(fFrameTime)

		if len(self._deqFrameTimes) > 0:
			fAvgFrameTime = sum(self._deqFrameTimes) / len(self._deqFrameTimes)
			fFps = 1.0 / fAvgFrameTime if fAvgFrameTime > 0 else 0
			self._statusBar.showMessage(
				f"FPS: {fFps:.1f} | 情绪: {sEmotion} ({fConfidence:.1f}%) | "
				f"模型: {self._sSelectedModelName}"
			)

	def _ProcessFrame(self, frame):
		"""
		Handle incoming frame from camera/video.
		Submits frame to background processor thread.

		Args:
			frame: Input frame.
		"""
		# Calculate display FPS (all incoming frames)
		self._iDisplayFrameCount += 1

		# Submit to processor thread (non-blocking)
		if self._processorThread and self._processorThread.isRunning():
			self._processorThread.addFrame(frame)

	def _ProcessStaticImage(self, frame):
		"""
		Process static image for emotion recognition.

		Args:
			frame: Input image.
		"""
		with self._processLock:
			displayFrame = None
			try:
				if frame is None or frame.size == 0:
					logger.warning("Received invalid frame")
					return

				displayFrame = frame.copy()

				# Face detection
				lstFaces = []
				try:
					lstFaces = self._detector.detect_faces(frame)
				except Exception as e:
					logger.warning(f"Face detection failed: {str(e)}")

				# Process each face
				for (iX1, iY1, iX2, iY2, fDetConf) in lstFaces:
					# Clamp coordinates
					iH, iW = frame.shape[:2]
					iX1 = max(0, min(int(iX1), iW - 1))
					iY1 = max(0, min(int(iY1), iH - 1))
					iX2 = max(0, min(int(iX2), iW - 1))
					iY2 = max(0, min(int(iY2), iH - 1))

					if iX2 <= iX1 or iY2 <= iY1:
						continue

					# Face alignment
					rect = dlib.rectangle(iX1, iY1, iX2, iY2)
					try:
						lstLandmarks = self._aligner.detect_landmarks(frame, [rect])
						if not lstLandmarks:
							continue

						alignedFace = self._aligner.align_face(frame, lstLandmarks[0])
						if alignedFace is None:
							continue

						# Emotion prediction
						tensor = ModelHelper.PreprocessImage(alignedFace)
						tensor = tensor.to(self._sDevice)

						with torch.no_grad():
							output = self._model(tensor)

						# Postprocess predictions
						sEmotion, fConf = ModelHelper.PostprocessPredictions(
							output[0], self._model.GetEmotionLabels()
						)

						# Draw result
						displayFrame = SafeDrawEmotionResult(
							displayFrame, sEmotion, fConf, (iX1, iY1, iX2, iY2)
						)

						# Add record
						self._AddRecord(sEmotion, fConf)

						# Update emotion history
						try:
							iIdx = self.CLST_EMOTION_LABELS.index(sEmotion)
						except ValueError:
							iIdx = 6
						self._deqEmotionHistory.append(iIdx)
						self._bDataChanged = True

						self._statusBar.showMessage(
							f"情绪: {sEmotion} ({fConf:.1f}%) | "
							f"模型: {self._sSelectedModelName}"
						)

						break  # Only process first face

					except Exception as e:
						logger.warning(f"Failed to process face: {str(e)}")
						continue

				# Update display
				if displayFrame is not None:
					self._imageDisplay.set_image(displayFrame)

				# Stop processing for static image
				self._StopProcessing()

			except Exception as e:
				logger.error(f"Critical error processing frame: {str(e)}")
				import traceback
				logger.error(traceback.format_exc())

	def _OpenCamera(self):
		"""Open camera for real-time emotion recognition."""
		# Enforce model selection before any recognition action
		if not self._bModelSelected:
			print("[警告] 未选择模型，请先选择一个模型后再打开摄像头")
			return

		logger.info("User clicked open camera")
		self._StopProcessing()
		self._sCurrentMode = "camera"
		self._iRecordCount = 0
		self._fLastFrameTime = time.time()
		self._deqFrameTimes.clear()
		self._iDisplayFrameCount = 0

		# Start processor thread
		if not self._processorThread.isRunning():
			self._processorThread.start()

		# Start camera thread
		self._cameraThread = CameraThread(source=0)
		self._cameraThread.frame_ready.connect(self._ProcessFrame)
		self._cameraThread.error_occurred.connect(self._OnError)
		self._cameraThread.start()

		self._bIsProcessing = True
		self._statusBar.showMessage(f"摄像头已开启 | 模型: {self._sSelectedModelName}")
		logger.info("Camera thread started")

	def _OpenImage(self):
		"""Open single image file."""
		# Enforce model selection before any recognition action
		if not self._bModelSelected:
			print("[警告] 未选择模型，请先选择一个模型后再选择图片")
			return

		logger.info("User clicked select image")
		sFilePath, _ = QFileDialog.getOpenFileName(
			self, "选择图片", "",
			"Image Files (*.png *.jpg *.jpeg *.bmp)"
		)

		if not sFilePath:
			return

		self._StopProcessing()
		self._sCurrentMode = "image"
		logger.info(f"Processing image: {sFilePath}")

		try:
			frame = cv2.imread(sFilePath)
			if frame is None:
				QMessageBox.warning(self, "错误", "无法读取图片文件")
				return

			self._bIsProcessing = True
			self._ProcessStaticImage(frame)
			logger.info("Image processing completed")

		except Exception as e:
			logger.error(f"Error processing image: {str(e)}")
			QMessageBox.warning(self, "错误", f"处理图片时出错:\n{str(e)}")

	def _OpenVideo(self):
		"""Open video file for emotion recognition."""
		# Enforce model selection before any recognition action
		if not self._bModelSelected:
			print("[警告] 未选择模型，请先选择一个模型后再选择视频")
			return

		logger.info("User clicked select video")
		sFilePath, _ = QFileDialog.getOpenFileName(
			self, "选择视频", "",
			"Video Files (*.mp4 *.avi *.mov)"
		)

		if not sFilePath:
			return

		self._StopProcessing()
		self._sCurrentMode = "video"
		self._iRecordCount = 0
		self._fLastFrameTime = time.time()
		self._deqFrameTimes.clear()
		self._iDisplayFrameCount = 0

		# Start processor thread
		if not self._processorThread.isRunning():
			self._processorThread.start()

		# Start video thread
		self._videoThread = CameraThread(source=sFilePath)
		self._videoThread.frame_ready.connect(self._ProcessFrame)
		self._videoThread.error_occurred.connect(self._OnError)
		self._videoThread.start()

		self._bIsProcessing = True
		self._statusBar.showMessage(
			f"正在播放视频: {os.path.basename(sFilePath)} | 模型: {self._sSelectedModelName}"
		)
		logger.info(f"Video thread started: {sFilePath}")

	def _OnError(self, sErrorMsg):
		"""
		Handle thread error.

		Args:
			sErrorMsg: Error message from thread.
		"""
		logger.error(f"Thread error: {sErrorMsg}")
		QMessageBox.warning(self, "错误", sErrorMsg)
		self._StopProcessing()

	def _StopProcessing(self):
		"""Stop all processing threads."""
		logger.info("User clicked stop button")

		if self._cameraThread:
			self._cameraThread.stop()
			self._cameraThread = None

		if self._videoThread:
			self._videoThread.stop()
			self._videoThread = None

		# Stop processor thread
		if self._processorThread and self._processorThread.isRunning():
			self._processorThread.stop()

		# Flush remaining records
		self._FlushRecords()
		self._recordTimer.stop()

		self._imageDisplay.clear()
		self._bIsProcessing = False

		sStatus = "已停止"
		if self._sSelectedModelName:
			sStatus += f" | 当前模型: {self._sSelectedModelName}"
		else:
			sStatus += " | 请先选择模型"
		self._statusBar.showMessage(sStatus)
		logger.info("All processing threads stopped")

	def _ShowAbout(self):
		"""Show about dialog."""
		QMessageBox.about(
			self, "关于",
			"基于深度学习的人脸情绪识别系统\n"
			"开发：肖晓伟\n"
			"指导老师：唐荣\n"
			"西南科技大学 信息与控制工程学院\n"
			"使用技术：PyTorch, PyQt5, OpenCV, Dlib"
		)

	def closeEvent(self, event):
		"""Handle window close event."""
		logger.info("Closing window, stopping all threads")
		self._StopProcessing()
		self._curveTimer.stop()
		event.accept()


def main():
	"""Main entry point."""
	app = QApplication(sys.argv)
	app.setApplicationName("Emotion Recognition GUI")
	app.setApplicationVersion("1.0")

	try:
		window = EmotionRecognitionGUI()
		window.show()
		logger.info("Application started successfully")
		sys.exit(app.exec_())

	except Exception as e:
		logger.critical(f"Application startup failed: {str(e)}")
		import traceback
		logger.critical(traceback.format_exc())
		QMessageBox.critical(None, "启动失败", f"应用程序启动失败:\n{str(e)}")
		sys.exit(1)


if __name__ == "__main__":
	main()