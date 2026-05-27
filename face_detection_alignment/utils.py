# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Utility functions for face detection and alignment module.
Contains image preprocessing, bounding box handling, and common operations.
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


def draw_bounding_box(image, x1, y1, x2, y2, color=(0, 255, 0), thickness=2, label=None):
	"""
	Draw bounding box on image with optional label.

	Args:
		image (numpy.ndarray): Input image.
		x1, y1, x2, y2: Bounding box coordinates.
		color (tuple): Bounding box color (BGR).
		thickness (int): Bounding box line thickness.
		label (str): Label text to display.

	Returns:
		numpy.ndarray: Image with bounding box drawn.
	"""
	output = image.copy()

	# Draw bounding box
	cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

	# Draw label background and text
	if label:
		text_size, baseline = cv2.getTextSize(
			label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
		)
		cv2.rectangle(
			output,
			(x1, y1 - text_size[1] - 5),
			(x1 + text_size[0], y1),
			color, -1
		)
		cv2.putText(
			output, label, (x1, y1 - 5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
		)

	return output


def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=2):
	"""
	Draw face landmarks on image.

	Args:
		image (numpy.ndarray): Input image.
		landmarks (numpy.ndarray): Landmark coordinates.
		color (tuple): Landmark color (BGR).
		radius (int): Landmark radius.

	Returns:
		numpy.ndarray: Image with landmarks drawn.
	"""
	output = image.copy()

	for (x, y) in landmarks:
		cv2.circle(output, (x, y), radius, color, -1)

	return output


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
	"""
	Resize image while maintaining aspect ratio.

	Args:
		image (numpy.ndarray): Input image.
		width (int): Target width (None to auto-calculate).
		height (int): Target height (None to auto-calculate).
		inter: Interpolation method.

	Returns:
		numpy.ndarray: Resized image.
	"""
	i_h, i_w = image.shape[:2]

	# Return original if no dimensions specified
	if width is None and height is None:
		return image

	# Calculate scale ratio
	if width is None:
		f_r = height / float(i_h)
		dim = (int(i_w * f_r), height)
	else:
		f_r = width / float(i_w)
		dim = (width, int(i_h * f_r))

	return cv2.resize(image, dim, interpolation=inter)


def rotate_image(image, angle, center=None, scale=1.0):
	"""
	Rotate image by specified angle.

	Args:
		image (numpy.ndarray): Input image.
		angle (float): Rotation angle in degrees.
		center: Rotation center point.
		scale (float): Scale factor.

	Returns:
		numpy.ndarray: Rotated image.
	"""
	i_h, i_w = image.shape[:2]

	if center is None:
		center = (i_w // 2, i_h // 2)

	# Calculate rotation matrix
	M = cv2.getRotationMatrix2D(center, angle, scale)

	# Calculate new bounding box size
	f_cos = np.abs(M[0, 0])
	f_sin = np.abs(M[0, 1])
	i_nW = int((i_h * f_sin) + (i_w * f_cos))
	i_nH = int((i_h * f_cos) + (i_w * f_sin))

	# Adjust rotation matrix for translation
	M[0, 2] += (i_nW / 2) - center[0]
	M[1, 2] += (i_nH / 2) - center[1]

	return cv2.warpAffine(image, M, (i_nW, i_nH))


def normalize_image(image):
	"""
	Normalize image using ImageNet statistics.

	Args:
		image (numpy.ndarray): Input image (0-255).

	Returns:
		numpy.ndarray: Normalized image.
	"""
	# Convert to float
	normalized = image.astype(np.float32)

	# Normalize to [0, 1]
	normalized = normalized / 255.0

	# Apply ImageNet normalization
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	normalized = (normalized - mean) / std

	return normalized


def denormalize_image(normalized_image):
	"""
	Denormalize image from ImageNet statistics.

	Args:
		normalized_image (numpy.ndarray): Normalized image.

	Returns:
		numpy.ndarray: Denormalized image (0-255 uint8).
	"""
	# Reverse ImageNet normalization
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	image = normalized_image * std + mean

	# Convert back to [0, 255] range
	image = np.clip(image, 0, 1)
	image = (image * 255).astype(np.uint8)

	return image


def crop_image(image, x1, y1, x2, y2):
	"""
	Crop image to specified region.

	Args:
		image (numpy.ndarray): Input image.
		x1, y1, x2, y2: Crop region coordinates.

	Returns:
		numpy.ndarray: Cropped image.
	"""
	i_h, i_w = image.shape[:2]

	# Clamp coordinates to valid range
	x1 = max(0, min(x1, i_w - 1))
	y1 = max(0, min(y1, i_h - 1))
	x2 = max(0, min(x2, i_w - 1))
	y2 = max(0, min(y2, i_h - 1))

	# Validate ordering
	if x1 > x2 or y1 > y2:
		logger.warning("Invalid crop coordinates, returning original image")
		return image

	return image[y1:y2 + 1, x1:x2 + 1]


def load_image(image_path):
	"""
	Load image from file path.

	Args:
		image_path (str): Path to image file.

	Returns:
		numpy.ndarray: Image data (BGR format).

	Raises:
		FileNotFoundError: If file does not exist.
		IOError: If file cannot be read.
	"""
	if not os.path.exists(image_path):
		raise FileNotFoundError(f"Image file not found: {image_path}")

	# Load image using cv2.imread (supports PNG, JPG, etc.)
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)

	if image is None:
		raise IOError(f"Failed to read image file: {image_path}")

	logger.debug(f"Image loaded: {image_path}, shape: {image.shape}")
	return image


def save_image(image, save_path):
	"""
	Save image to file path.

	Args:
		image (numpy.ndarray): Image data.
		save_path (str): Path to save image.
	"""
	# Ensure save directory exists
	save_dir = os.path.dirname(save_path)
	if save_dir and not os.path.exists(save_dir):
		os.makedirs(save_dir, exist_ok=True)

	try:
		# Save image (format determined by extension)
		cv2.imwrite(save_path, image)
		logger.debug(f"Image saved: {save_path}")
	except Exception as e:
		logger.error(f"Failed to save image: {str(e)}")


def calculate_iou(box1, box2):
	"""
	Calculate Intersection over Union (IoU) between two bounding boxes.

	Args:
		box1: First bounding box (x1, y1, x2, y2).
		box2: Second bounding box (x1, y1, x2, y2).

	Returns:
		float: IoU value.
	"""
	x1a, y1a, x2a, y2a = box1
	x1b, y1b, x2b, y2b = box2

	# Calculate intersection area
	x_left = max(x1a, x1b)
	y_top = max(y1a, y1b)
	x_right = min(x2a, x2b)
	y_bottom = min(y2a, y2b)

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	f_intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

	# Calculate union area
	f_box1_area = (x2a - x1a + 1) * (y2a - y1a + 1)
	f_box2_area = (x2b - x1b + 1) * (y2b - y1b + 1)
	f_union_area = f_box1_area + f_box2_area - f_intersection_area

	return f_intersection_area / max(f_union_area, 1.0)


def non_max_suppression(boxes, overlap_thresh):
	"""
	Apply non-maximum suppression to bounding boxes.

	Args:
		boxes: Bounding boxes as numpy array (N, 5) with (x1, y1, x2, y2, score).
		overlap_thresh: Overlap threshold for suppression.

	Returns:
		numpy.ndarray: Suppressed bounding boxes.
	"""
	# Handle empty input
	if len(boxes) == 0:
		return np.array([])

	# Convert to numpy array if needed
	if isinstance(boxes, list):
		boxes = np.array(boxes)

	# Ensure float type for computation
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# Handle single box
	if len(boxes) == 1:
		return boxes.astype("int")

	pick = []

	# Extract coordinates
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	scores = boxes[:, 4]

	# Calculate areas
	area = (x2 - x1 + 1) * (y2 - y1 + 1)

	# Sort by confidence score
	idxs = np.argsort(scores)

	while len(idxs) > 0:
		i_last = len(idxs) - 1
		i = idxs[i_last]
		pick.append(i)

		# Find overlap regions
		xx1 = np.maximum(x1[i], x1[idxs[:i_last]])
		yy1 = np.maximum(y1[i], y1[idxs[:i_last]])
		xx2 = np.minimum(x2[i], x2[idxs[:i_last]])
		yy2 = np.minimum(y2[i], y2[idxs[:i_last]])

		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		overlap = (w * h) / area[idxs[:i_last]]

		# Remove overlapping boxes
		idxs = np.delete(
			idxs,
			np.concatenate(([i_last], np.where(overlap > overlap_thresh)[0]))
		)

	return boxes[pick].astype("int")