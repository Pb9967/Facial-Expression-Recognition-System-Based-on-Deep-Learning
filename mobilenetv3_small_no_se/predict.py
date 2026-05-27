# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Emotion recognition inference script (MobileNetV3-Small NO SE).
Provides image, video and camera-based emotion prediction.
Ablation baseline: remove SE Block attention mechanism.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sProjectRoot = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(sProjectRoot))

import cv2
import torch
import numpy as np
import dlib

from mobilenetv3_small_no_se.config import ConfigLoader
from mobilenetv3_small_no_se.model import EmotionNet
from mobilenetv3_small_no_se.utils import ModelHelper
from mobilenetv3_small_no_se.utils import VisualizationHelper
from face_detection_alignment.detector import FaceDetector
from face_detection_alignment.aligner import FaceAligner
from face_detection_alignment.utils import load_image
from face_detection_alignment.utils import save_image


class PredictEmotion:
	"""
	Emotion prediction controller.
	Handles model loading and inference for various input types.
	"""

	def __init__(self, dctConfig):
		"""
		Initialize prediction controller.

		Args:
			dctConfig (dict): Configuration dictionary.
		"""
		self._dctConfig = dctConfig
		self._sDevice = dctConfig.get("sDevice", "cpu")
		self._model = None
		self._detector = None
		self._aligner = None
		self._lstEmotionLabels = None

	def LoadModel(self, sWeightPath=None):
		"""
		Load emotion recognition model.

		Args:
			sWeightPath (str): Path to model weights.

		Raises:
			FileNotFoundError: If weight file not found.
		"""
		# Create model instance
		self._model = EmotionNet(self._dctConfig)
		self._lstEmotionLabels = self._model.GetEmotionLabels()

		# Determine weight path
		if sWeightPath and os.path.exists(sWeightPath):
			pass
		else:
			# Use default weight path
			sDefaultPath = self._GetDefaultWeightPath()
			if os.path.exists(sDefaultPath):
				sWeightPath = sDefaultPath
			else:
				raise FileNotFoundError("Weight file not found")

		# Load weights
		self._model = ModelHelper.LoadWeights(self._model, sWeightPath)
		self._model.to(self._sDevice)
		self._model.eval()

	def _GetDefaultWeightPath(self):
		"""
		Get default weight file path.

		Returns:
			str: Default weight path.
		"""
		sModelDir = Path(__file__).parent
		return str(
			sModelDir / "train_120p" / "weights" / "final_model.pth"
		)

	def InitDetectors(self):
		"""Initialize face detector and aligner."""
		self._detector = FaceDetector()
		self._aligner = FaceAligner()

	def PredictImage(self, sImagePath):
		"""
		Predict emotion for a single image.

		Args:
			sImagePath (str): Path to input image.

		Returns:
			np.ndarray: Result image with annotations, or None if no face.
		"""
		# Load image
		image = load_image(sImagePath)

		# Detect faces
		lstFaces = self._detector.detect_faces(image)

		if not lstFaces:
			print("No faces detected")
			return None

		# Process each face
		lstEmotions = []
		lstConfidences = []

		for tplFace in lstFaces:
			iX1, iY1, iX2, iY2, fDetConf = tplFace

			# Detect landmarks
			lstLandmarks = self._aligner.detect_landmarks(
				image,
				[dlib.rectangle(iX1, iY1, iX2, iY2)]
			)

			if lstLandmarks:
				# Align face
				alignedFace = self._aligner.align_face(
					image, lstLandmarks[0]
				)

				if alignedFace is not None:
					# Predict emotion
					sEmotion, fConf = self._PredictFace(alignedFace)
					lstEmotions.append(sEmotion)
					lstConfidences.append(fConf)

		# Visualize results
		resultImage = VisualizationHelper.VisualizePredictions(
			image, lstEmotions, lstConfidences, lstFaces
		)

		return resultImage

	def _PredictFace(self, faceImage):
		"""
		Predict emotion for a single aligned face.

		Args:
			faceImage (np.ndarray): Aligned face image.

		Returns:
			tuple: (emotion_label, confidence)
		"""
		# Preprocess
		tensor = ModelHelper.PreprocessImage(faceImage)
		tensor = tensor.to(self._sDevice)

		# Inference
		with torch.no_grad():
			predictions = self._model(tensor)

		# Postprocess
		sEmotion, fConf = ModelHelper.PostprocessPredictions(
			predictions[0], self._lstEmotionLabels
		)

		return sEmotion, fConf

	def PredictVideo(self, sVideoPath, sOutputPath=None):
		"""
		Predict emotions in video file.

		Args:
			sVideoPath (str): Path to input video.
			sOutputPath (str): Path to output video.

		Returns:
			str: Path to output video.
		"""
		# Open video
		cap = cv2.VideoCapture(sVideoPath)

		if not cap.isOpened():
			print("Cannot open video")
			return None

		# Get video properties
		iWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		iHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		iFps = int(cap.get(cv2.CAP_PROP_FPS))
		iTotalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		# Setup output
		if sOutputPath is None:
			sOutputPath = "output_video.mp4"

		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		out = cv2.VideoWriter(sOutputPath, fourcc, iFps, (iWidth, iHeight))

		iFrameCount = 0

		while cap.isOpened():
			bRet, frame = cap.read()

			if not bRet:
				break

			iFrameCount += 1
			print(f"Processing frame: {iFrameCount}/{iTotalFrames}")

			# Process frame
			resultFrame = self._ProcessFrame(frame)

			# Write output
			out.write(resultFrame)

		# Cleanup
		cap.release()
		out.release()

		print(f"Output saved to: {sOutputPath}")
		return sOutputPath

	def _ProcessFrame(self, frame):
		"""
		Process single video frame.

		Args:
			frame (np.ndarray): Input frame.

		Returns:
			np.ndarray: Processed frame with annotations.
		"""
		lstFaces = self._detector.detect_faces(frame)
		lstEmotions = []
		lstConfidences = []

		for tplFace in lstFaces:
			iX1, iY1, iX2, iY2, fDetConf = tplFace

			lstLandmarks = self._aligner.detect_landmarks(
				frame,
				[dlib.rectangle(iX1, iY1, iX2, iY2)]
			)

			if lstLandmarks:
				alignedFace = self._aligner.align_face(
					frame, lstLandmarks[0]
				)

				if alignedFace is not None:
					sEmotion, fConf = self._PredictFace(alignedFace)
					lstEmotions.append(sEmotion)
					lstConfidences.append(fConf)

		resultFrame = VisualizationHelper.VisualizePredictions(
			frame, lstEmotions, lstConfidences, lstFaces
		)

		return resultFrame

	def PredictCamera(self):
		"""
		Real-time camera emotion recognition.
		Press 'q' to quit.
		"""
		cap = cv2.VideoCapture(0)

		if not cap.isOpened():
			print("Cannot open camera")
			return

		print("Camera started. Press 'q' to quit.")

		while True:
			bRet, frame = cap.read()

			if not bRet:
				print("Cannot read frame")
				break

			# Process frame
			resultFrame = self._ProcessFrame(frame)

			# Display
			cv2.imshow("Emotion Recognition", resultFrame)

			# Check quit
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		# Cleanup
		cap.release()
		cv2.destroyAllWindows()


def main():
	"""Main entry point for prediction."""
	# Training epochs used for the model to load
	# Modify this value to switch between different training runs
	iTrainEpochs = 2

	# Load configuration
	configLoader = ConfigLoader()
	dctConfig = configLoader.GetConfig()

	# Update configuration
	dctConfig.update({
		"sInputType": "camera",
		"sInputPath": str(
			Path(__file__).parent.parent.parent / "test_images" / "face_2.png"
		),
		"sOutputPath": str(
			Path(__file__).parent / "outputs" / "test_images_face_2.jpg"
		),
		"sWeightPath": str(
			Path(__file__).parent / f"train_{iTrainEpochs}p" / "weights" / "final_model.pth"
		),
		"sDevice": "cuda" if torch.cuda.is_available() else "cpu"
	})

	# Print configuration
	print("=== Emotion Recognition (Small NO SE) ===")
	print("Configuration:")
	for sKey, value in dctConfig.items():
		print(f"  {sKey}: {value}")
	print()

	# Ensure output directory exists
	if dctConfig["sInputType"] != "camera":
		sOutputDir = os.path.dirname(dctConfig["sOutputPath"])
		os.makedirs(sOutputDir, exist_ok=True)

	try:
		# Initialize predictor
		predictor = PredictEmotion(dctConfig)

		# Load model
		print("Loading model...")
		predictor.LoadModel(dctConfig.get("sWeightPath"))
		print("Model loaded successfully")
		print()

		# Initialize detectors
		print("Initializing detectors...")
		predictor.InitDetectors()
		print("Detectors initialized")
		print()

		# Process based on input type
		sInputType = dctConfig["sInputType"]

		if sInputType == "image":
			_ProcessImage(predictor, dctConfig)
		elif sInputType == "video":
			_ProcessVideo(predictor, dctConfig)
		elif sInputType == "camera":
			_ProcessCamera(predictor)
		else:
			print(f"Unsupported input type: {sInputType}")
			print("Supported types: image, video, camera")

	except Exception as e:
		print(f"Error: {str(e)}")
		import traceback
		print("\nDetailed error:")
		print(traceback.format_exc())


def _ProcessImage(predictor, dctConfig):
	"""Process image input."""
	sInputPath = dctConfig["sInputPath"]
	sOutputPath = dctConfig["sOutputPath"]

	if not os.path.exists(sInputPath):
		print(f"Error: Image not found: {sInputPath}")
		return

	print(f"Processing image: {sInputPath}")
	resultImage = predictor.PredictImage(sInputPath)

	if resultImage is None:
		print("No faces detected")
	else:
		save_image(resultImage, sOutputPath)
		print(f"Result saved to: {sOutputPath}")

		# Display result
		cv2.imshow("Emotion Recognition", resultImage)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


def _ProcessVideo(predictor, dctConfig):
	"""Process video input."""
	sInputPath = dctConfig["sInputPath"]
	sOutputPath = dctConfig["sOutputPath"]

	if not os.path.exists(sInputPath):
		print(f"Error: Video not found: {sInputPath}")
		return

	print(f"Processing video: {sInputPath}")
	sResultPath = predictor.PredictVideo(sInputPath, sOutputPath)
	print(f"Result saved to: {sResultPath}")


def _ProcessCamera(predictor):
	"""Process camera input."""
	print("Starting camera real-time detection...")
	print("Press 'q' to quit")
	predictor.PredictCamera()


if __name__ == "__main__":
	main()