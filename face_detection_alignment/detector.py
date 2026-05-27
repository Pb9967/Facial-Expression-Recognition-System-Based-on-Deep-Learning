# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Face detection module using OpenCV DNN with Caffe SSD model.
Supports image and video stream input.
"""

import os
import cv2
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
	'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


class FaceDetector:
	"""
	Face detector class using OpenCV DNN module.
	Uses Caffe SSD model for efficient face detection.
	"""

	def __init__(self, model_dir=None):
		"""
		Initialize face detector.

		Args:
			model_dir (str): Directory containing model files.
		"""
		if model_dir is None:
			model_dir = os.path.join(
				os.path.dirname(os.path.abspath(__file__)),
				"models"
			)

		self.s_model_dir = model_dir
		self.net = None
		self.f_confidence_threshold = 0.7

		# Load face detection model
		self._load_model()

	def _load_model(self):
		"""
		Load OpenCV DNN face detection model.
		Uses Caffe model with high detection accuracy.
		"""
		# Model file paths
		s_prototxt_path = os.path.join(
			self.s_model_dir, "deploy.prototxt"
		)
		s_model_path = os.path.join(
			self.s_model_dir,
			"res10_300x300_ssd_iter_140000.caffemodel"
		)

		# Check if model files exist
		if not os.path.exists(s_prototxt_path) or not os.path.exists(s_model_path):
			logger.error("Face detection model files not found")
			logger.error(f"Expected paths: {s_prototxt_path} and {s_model_path}")
			raise FileNotFoundError("Face detection model files not found")

		try:
			# Load model
			self.net = cv2.dnn.readNetFromCaffe(s_prototxt_path, s_model_path)
			logger.info("Face detection model loaded successfully")
		except Exception as e:
			logger.error(f"Failed to load face detection model: {str(e)}")
			raise

	def detect_faces(self, image, scale_factor=1.0, min_confidence=None):
		"""
		Detect faces in image.

		Args:
			image (numpy.ndarray): Input image (BGR format).
			scale_factor (float): Image scale factor.
			min_confidence (float): Minimum confidence threshold.

		Returns:
			list: List of face bounding boxes as (x1, y1, x2, y2, confidence).
		"""
		if min_confidence is None:
			min_confidence = self.f_confidence_threshold

		# Handle invalid input
		if image is None or image.size == 0:
			logger.warning("Empty image provided for face detection")
			return []

		# Scale image if needed
		if scale_factor != 1.0:
			image = cv2.resize(
				image, None,
				fx=scale_factor, fy=scale_factor,
				interpolation=cv2.INTER_AREA
			)

		# Get image dimensions
		i_h, i_w = image.shape[:2]

		# Construct input blob for DNN
		blob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)),
			1.0, (300, 300), (104.0, 177.0, 123.0)
		)

		# Forward pass to get detections
		self.net.setInput(blob)
		detections = self.net.forward()

		faces = []

		# Parse detection results
		for i in range(detections.shape[2]):
			# Get confidence
			f_confidence = detections[0, 0, i, 2]

			if f_confidence > min_confidence:
				# Calculate bounding box coordinates
				box = detections[0, 0, i, 3:7] * np.array([i_w, i_h, i_w, i_h])
				(i_startX, i_startY, i_endX, i_endY) = box.astype("int")

				# Validate and clamp bounding box
				b_valid, i_x1, i_y1, i_x2, i_y2 = self._validate_bounding_box(
					i_startX, i_startY, i_endX, i_endY, i_w, i_h
				)

				if b_valid:
					faces.append((i_x1, i_y1, i_x2, i_y2, f_confidence))

		# Sort by confidence in descending order
		faces = sorted(faces, key=lambda x: x[4], reverse=True)

		logger.debug(f"Detected {len(faces)} faces")
		return faces

	def _validate_bounding_box(self, x1, y1, x2, y2, img_width, img_height):
		"""
		Validate and clamp bounding box coordinates.

		Args:
			x1, y1, x2, y2: Bounding box coordinates.
			img_width (int): Image width.
			img_height (int): Image height.

		Returns:
			tuple: (is_valid, clamped_x1, clamped_y1, clamped_x2, clamped_y2)
		"""
		# Clamp coordinates to image bounds
		x1 = max(0, min(x1, img_width - 1))
		y1 = max(0, min(y1, img_height - 1))
		x2 = max(0, min(x2, img_width - 1))
		y2 = max(0, min(y2, img_height - 1))

		# Ensure correct ordering (x1 < x2, y1 < y2)
		if x1 > x2:
			x1, x2 = x2, x1
		if y1 > y2:
			y1, y2 = y2, y1

		# Minimum size check
		i_min_size = 20
		if (x2 - x1) < i_min_size or (y2 - y1) < i_min_size:
			return False, 0, 0, 0, 0

		# Aspect ratio check (avoid extreme ratios)
		f_aspect_ratio = (x2 - x1) / max(y2 - y1, 1)
		if f_aspect_ratio < 0.5 or f_aspect_ratio > 2.0:
			return False, 0, 0, 0, 0

		return True, x1, y1, x2, y2

	def draw_detection_results(self, image, faces, color=(0, 255, 0), thickness=2):
		"""
		Draw detection results on image.

		Args:
			image (numpy.ndarray): Input image (BGR format).
			faces (list): List of face bounding boxes.
			color (tuple): Bounding box color (BGR).
			thickness (int): Bounding box line thickness.

		Returns:
			numpy.ndarray: Image with detection results drawn.
		"""
		output = image.copy()

		for (x1, y1, x2, y2, f_confidence) in faces:
			# Draw bounding box
			cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

			# Draw confidence score
			s_text = f"{f_confidence:.2f}"
			cv2.putText(
				output, s_text, (x1, y1 - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
			)

		return output

	def set_confidence_threshold(self, threshold):
		"""
		Set minimum confidence threshold.

		Args:
			threshold (float): Confidence threshold (0.0-1.0).
		"""
		if 0.0 <= threshold <= 1.0:
			self.f_confidence_threshold = threshold
			logger.info(f"Confidence threshold set to: {threshold}")
		else:
			logger.warning(
				f"Confidence threshold should be in [0.0, 1.0], got: {threshold}"
			)