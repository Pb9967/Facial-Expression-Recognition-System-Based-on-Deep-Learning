# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Image display component.
Optimized for performance with automatic scaling.
"""

import cv2
import numpy as np
import logging

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy

logger = logging.getLogger(__name__)


class ImageWidget(QLabel):
	"""
	Optimized image display widget.
	Supports automatic scaling with aspect ratio preservation.
	"""

	def __init__(self, parent=None):
		"""
		Initialize image display widget.

		Args:
			parent: Parent widget.
		"""
		super().__init__(parent)
		self.setAlignment(Qt.AlignCenter)
		self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.setMinimumSize(100, 100)
		self._currentPixmap = None
		self._bHighQuality = False  # Use fast transformation by default

	def set_image(self, cvImage):
		"""
		Set image to display from OpenCV format.

		Args:
			cvImage (np.ndarray): BGR format image.
		"""
		try:
			# Validate input
			if cvImage is None or not isinstance(cvImage, np.ndarray):
				logger.warning("Invalid input image")
				self.clear()
				return

			if cvImage.size == 0:
				logger.warning("Empty input image")
				self.clear()
				return

			if cvImage.shape[0] == 0 or cvImage.shape[1] == 0:
				logger.warning("Invalid image dimensions")
				self.clear()
				return

			# Convert BGR to RGB
			if len(cvImage.shape) == 3 and cvImage.shape[2] == 3:
				imgRgb = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
			elif len(cvImage.shape) == 3 and cvImage.shape[2] == 4:
				imgRgb = cv2.cvtColor(cvImage, cv2.COLOR_BGRA2RGB)
			else:
				imgRgb = cvImage

			# Convert to QImage
			qimage = self._CvToQImage(imgRgb)
			if qimage is None:
				logger.warning("Failed to convert image format")
				return

			# Create QPixmap
			pixmap = QPixmap.fromImage(qimage)
			if pixmap.isNull():
				logger.warning("Failed to create QPixmap")
				return

			# Store and display
			self._currentPixmap = pixmap
			self._UpdateDisplay()

		except Exception as e:
			logger.error(f"Failed to set image: {str(e)}")
			import traceback
			logger.error(traceback.format_exc())
			self.clear()

	def _CvToQImage(self, cvImage):
		"""
		Safely convert OpenCV image to QImage.

		Args:
			cvImage (np.ndarray): Input image.

		Returns:
			QImage: Converted image, or None on error.
		"""
		try:
			iH, iW = cvImage.shape[:2]

			if len(cvImage.shape) == 2:
				# Grayscale image
				iBytesPerLine = iW
				qimage = QImage(
					cvImage.data, iW, iH, iBytesPerLine,
					QImage.Format_Grayscale8
				)
			elif len(cvImage.shape) == 3:
				if cvImage.shape[2] == 3:
					# RGB image
					iBytesPerLine = 3 * iW
					qimage = QImage(
						cvImage.data, iW, iH, iBytesPerLine,
						QImage.Format_RGB888
					)
				elif cvImage.shape[2] == 4:
					# RGBA image
					iBytesPerLine = 4 * iW
					qimage = QImage(
						cvImage.data, iW, iH, iBytesPerLine,
						QImage.Format_RGBA8888
					)
				else:
					logger.warning(f"Unsupported channel count: {cvImage.shape[2]}")
					return None
			else:
				logger.warning(f"Unsupported image shape: {cvImage.shape}")
				return None

			# Return copy to ensure data ownership
			return qimage.copy()

		except Exception as e:
			logger.error(f"Image conversion failed: {str(e)}")
			return None

	def _UpdateDisplay(self):
		"""Update display with scaled pixmap."""
		try:
			if self._currentPixmap is None or self._currentPixmap.isNull():
				return

			# Scale image to fit widget size
			if self.width() > 0 and self.height() > 0:
				# Use fast transformation for real-time display
				transformMode = (
					Qt.SmoothTransformation if self._bHighQuality
					else Qt.FastTransformation
				)
				scaledPixmap = self._currentPixmap.scaled(
					self.size(),
					Qt.KeepAspectRatio,
					transformMode
				)
				self.setPixmap(scaledPixmap)
			else:
				self.setPixmap(self._currentPixmap)

		except Exception as e:
			logger.error(f"Failed to update display: {str(e)}")

	def setHighQuality(self, bEnabled):
		"""
		Set high quality scaling mode.

		Args:
			bEnabled (bool): Enable high quality scaling.
		"""
		self._bHighQuality = bEnabled

	def clear(self):
		"""Clear displayed image."""
		try:
			self._currentPixmap = None
			super().clear()
		except Exception as e:
			logger.error(f"Failed to clear image: {str(e)}")

	def resizeEvent(self, event):
		"""Handle resize event to update display."""
		super().resizeEvent(event)
		if self._currentPixmap is not None:
			self._UpdateDisplay()

	def __del__(self):
		"""Ensure resource cleanup."""
		try:
			self.clear()
		except Exception:
			pass