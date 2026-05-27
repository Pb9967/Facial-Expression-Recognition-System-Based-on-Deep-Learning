# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Face alignment module using Dlib for landmark detection and alignment.
"""

import os
import cv2
import dlib
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class FaceAligner:
	"""
	Face aligner class using Dlib's 68-point landmark detector.
	"""

	def __init__(self, predictor_path=None):
		"""
		Initialize face aligner.

		Args:
			predictor_path (str): Path to shape predictor file.
		"""
		if predictor_path is None:
			predictor_path = os.path.join(
				os.path.dirname(os.path.abspath(__file__)),
				"models",
				"shape_predictor_68_face_landmarks.dat"
			)

		self.s_predictor_path = predictor_path
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = None

		# Load shape predictor
		self._load_predictor()

	def _load_predictor(self):
		"""
		Load shape predictor model file.
		"""
		if not os.path.exists(self.s_predictor_path):
			logger.error("Face landmark predictor file not found")
			logger.error(f"Expected path: {self.s_predictor_path}")
			raise FileNotFoundError(
				f"Shape predictor file not found: {self.s_predictor_path}"
			)

		try:
			self.predictor = dlib.shape_predictor(self.s_predictor_path)
			logger.info("Face landmark predictor loaded successfully")
		except Exception as e:
			logger.error(f"Failed to load face landmark predictor: {str(e)}")
			raise

	def _ensure_valid_image(self, image):
		"""
		Ensure image is valid for Dlib processing.
		Dlib requires 8-bit grayscale or RGB image (not RGBA, not float).

		Args:
			image (numpy.ndarray): Input image.

		Returns:
			numpy.ndarray: Valid image for Dlib.
		"""
		# Handle None or empty image
		if image is None or image.size == 0:
			raise ValueError("Image is None or empty")

		# Convert to uint8 if float
		if image.dtype != np.uint8:
			image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

		# Handle RGBA (4 channels) - convert to BGR
		if len(image.shape) == 3 and image.shape[2] == 4:
			image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
			logger.debug("Converted RGBA image to BGR")

		# Handle single channel (already grayscale)
		if len(image.shape) == 2:
			return np.ascontiguousarray(image)

		# Handle 3-channel image (BGR or RGB)
		if len(image.shape) == 3 and image.shape[2] == 3:
			return np.ascontiguousarray(image)

		raise ValueError(f"Unsupported image shape: {image.shape}")

	def detect_landmarks(self, image, faces=None):
		"""
		Detect face landmarks.

		Args:
			image (numpy.ndarray): Input image (BGR format).
			faces (list): Pre-detected face bounding boxes.

		Returns:
			list: List of landmarks, each containing 68 landmark coordinates.
		"""
		# Ensure image is valid for Dlib
		valid_image = self._ensure_valid_image(image)

		# Convert to grayscale for landmark detection
		if len(valid_image.shape) == 3:
			gray = cv2.cvtColor(valid_image, cv2.COLOR_BGR2GRAY)
		else:
			gray = valid_image

		# 关键修改：确保灰度图像内存连续且可写（dlib 要求）
		gray = np.ascontiguousarray(gray)

		# Detect face bounding boxes if not provided
		if faces is None:
			faces = self.detector(gray, 1)

		landmarks = []

		for face in faces:
			try:
				# Detect landmarks
				shape = self.predictor(gray, face)

				# Convert to numpy array
				landmarks.append(self._shape_to_np(shape))
			except Exception as e:
				logger.error(f"Failed to detect landmarks: {str(e)}")

		logger.debug(f"Detected {len(landmarks)} sets of landmarks")
		return landmarks

	def align_face(self, image, landmarks, output_size=(224, 224), margin=48):
		"""
		Align face using affine transformation based on eye positions.

		Args:
			image (numpy.ndarray): Input image (BGR format).
			landmarks (numpy.ndarray): Face landmark coordinates.
			output_size (tuple): Output image size (width, height).
			margin (int): Margin to add around the aligned face.

		Returns:
			numpy.ndarray: Aligned face image.
		"""
		try:
			# Ensure image is valid
			valid_image = self._ensure_valid_image(image)

			# Calculate affine transformation matrix
			M = self._calculate_affine_matrix(
				landmarks, output_size, margin
			)

			# Apply affine transformation
			aligned_face = cv2.warpAffine(
				valid_image, M, (output_size[0], output_size[1])
			)

			logger.debug("Face aligned successfully")
			return aligned_face
		except Exception as e:
			logger.error(f"Failed to align face: {str(e)}")
			return None

	def _calculate_affine_matrix(self, landmarks, output_size, margin):
		"""
		Calculate affine transformation matrix based on eye positions.
		The transformation aligns the face so eyes are horizontal and centered.

		Args:
			landmarks (numpy.ndarray): Face landmark coordinates (68 points).
			output_size (tuple): Output image size (width, height).
			margin (int): Margin around the face.

		Returns:
			numpy.ndarray: 2x3 affine transformation matrix.
		"""
		# Extract eye centers (landmarks 36-41: left eye, 42-47: right eye)
		left_eye = np.mean(landmarks[36:42], axis=0)
		right_eye = np.mean(landmarks[42:48], axis=0)

		# Calculate eye center
		eyes_center = (left_eye + right_eye) * 0.5

		# Calculate rotation angle (to make eyes horizontal)
		dx = right_eye[0] - left_eye[0]
		dy = right_eye[1] - left_eye[1]
		angle = np.degrees(np.arctan2(dy, dx))

		# Calculate scale factor based on desired eye distance
		# Target: eyes should span about 40% of output width
		f_desired_eye_dist = output_size[0] * 0.4
		f_actual_eye_dist = np.linalg.norm(right_eye - left_eye)
		f_scale = f_desired_eye_dist / max(f_actual_eye_dist, 1.0)

		# Get rotation matrix centered at eyes center
		M = cv2.getRotationMatrix2D(
			(eyes_center[0], eyes_center[1]), angle, f_scale
		)

		# Calculate translation to center the face in output image
		# After rotation, eyes center should be at (output_width/2, output_height/3)
		f_target_x = output_size[0] / 2
		f_target_y = output_size[1] / 3

		# Update translation in the transformation matrix
		M[0, 2] = f_target_x - eyes_center[0] * f_scale
		M[1, 2] = f_target_y - eyes_center[1] * f_scale

		return M

	def _shape_to_np(self, shape):
		"""
		Convert Dlib shape object to numpy array.

		Args:
			shape: Dlib shape object containing 68 landmarks.

		Returns:
			numpy.ndarray: Array of landmark coordinates (68, 2).
		"""
		coords = np.zeros((68, 2), dtype="int")
		for i in range(0, 68):
			coords[i] = (shape.part(i).x, shape.part(i).y)
		return coords

	def draw_landmarks(self, image, landmarks, color=(0, 255, 0), radius=2):
		"""
		Draw face landmarks on image.

		Args:
			image (numpy.ndarray): Input image (BGR format).
			landmarks (list): List of landmark arrays.
			color (tuple): Landmark color (BGR).
			radius (int): Landmark radius.

		Returns:
			numpy.ndarray: Image with landmarks drawn.
		"""
		output = image.copy()

		for face_landmarks in landmarks:
			for (x, y) in face_landmarks:
				cv2.circle(output, (x, y), radius, color, -1)

		logger.debug("Landmarks drawn successfully")
		return output

	def preprocess_face(self, image, face_detector=None, output_size=(224, 224), margin=48):
		"""
		Complete face preprocessing pipeline: detect -> align.

		Args:
			image (numpy.ndarray): Input image (BGR format).
			face_detector: Face detector instance.
			output_size (tuple): Output size.
			margin (int): Margin around face.

		Returns:
			list: List of aligned face images.
		"""
		aligned_faces = []

		try:
			# Ensure image is valid
			valid_image = self._ensure_valid_image(image)

			# Face detection
			if face_detector is None:
				gray = cv2.cvtColor(valid_image, cv2.COLOR_BGR2GRAY)
				faces = self.detector(gray, 1)
			else:
				faces = face_detector.detect_faces(valid_image)
				faces = [
					dlib.rectangle(x1, y1, x2, y2)
					for (x1, y1, x2, y2, _) in faces
				]

			# Landmark detection
			landmarks = self.detect_landmarks(valid_image, faces)

			# Face alignment
			for landmark in landmarks:
				aligned_face = self.align_face(
					valid_image, landmark, output_size, margin
				)
				if aligned_face is not None:
					aligned_faces.append(aligned_face)

			logger.debug(
				f"Preprocessing complete: {len(aligned_faces)} aligned faces"
			)

		except Exception as e:
			logger.error(f"Face preprocessing failed: {str(e)}")

		return aligned_faces