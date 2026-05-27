# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Camera thread component based on QThread.
Provides frame capture with thread-safe signal emission.
"""

import cv2
import numpy as np
import logging

from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class CameraThread(QThread):
	"""
	Safe camera thread for frame capture.
	Emits deep-copied frames to avoid cross-thread memory sharing.
	"""

	# Signal definitions
	frame_ready = pyqtSignal(np.ndarray)
	error_occurred = pyqtSignal(str)

	def __init__(self, source=0):
		"""
		Initialize camera thread.

		Args:
			source: Camera index (int) or video file path (str).
		"""
		super().__init__()
		self._source = source
		self._cap = None
		self._bRunning = False
		self._iFrameInterval = 33  # ~30 FPS

	def run(self):
		"""Thread main loop for frame capture."""
		# Select backend based on source type
		if isinstance(self._source, int) and self._source >= 0:
			# Use DirectShow for Windows camera
			self._cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
		else:
			# Video file
			self._cap = cv2.VideoCapture(self._source)

		# Check if opened successfully
		if not self._cap.isOpened():
			self.error_occurred.emit(f"无法打开源: {self._source}")
			return

		self._bRunning = True
		logger.info(f"Camera thread started, source: {self._source}")

		while self._bRunning:
			bRet, frame = self._cap.read()

			if not bRet:
				# Video file reached end
				if isinstance(self._source, str):
					logger.info("Video playback completed")
					break
				# Camera temporary failure, wait and retry
				self.msleep(100)
				continue

			# Emit deep copy to avoid cross-thread memory sharing
			self.frame_ready.emit(frame.copy())

			# Control frame rate
			self.msleep(self._iFrameInterval)

		# Release resources
		if self._cap:
			self._cap.release()
		logger.info("Camera thread stopped")

	def stop(self):
		"""Stop thread and release resources."""
		self._bRunning = False
		self.wait()  # Wait for thread to finish

	def setFrameRate(self, iFps):
		"""
		Set target frame rate.

		Args:
			iFps (int): Target frames per second.
		"""
		if iFps > 0:
			self._iFrameInterval = int(1000 / iFps)