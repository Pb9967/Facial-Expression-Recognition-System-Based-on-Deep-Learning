# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
GUI utility functions.
Provides safe drawing functions with boundary handling.
"""

import cv2
import numpy as np
import logging

# Configure unified logging
logger = logging.getLogger(__name__)


def SafeDrawEmotionResult(image, sEmotionLabel, fConfidence, tplFaceRect):
	"""
	Safely draw emotion recognition result on image with boundary handling.

	Args:
		image (np.ndarray): BGR format image.
		sEmotionLabel (str): Emotion label string.
		fConfidence (float): Confidence score (percentage).
		tplFaceRect: Face bounding box, supports (x1, y1, x2, y2) tuple.

	Returns:
		np.ndarray: Image with drawn results, or original image on error.
	"""
	try:
		# Validate input image
		if image is None or not isinstance(image, np.ndarray):
			logger.warning("Invalid input image")
			return None

		if image.size == 0:
			logger.warning("Empty input image")
			return None

		# Get image dimensions
		iH, iW = image.shape[:2]
		if iH == 0 or iW == 0:
			logger.warning("Invalid image dimensions")
			return None

		# Validate face rectangle format
		if not isinstance(tplFaceRect, (list, tuple)) or len(tplFaceRect) < 4:
			logger.warning(f"Invalid face rect format: {tplFaceRect}")
			return image

		# Parse face rectangle coordinates
		iX1, iY1, iX2, iY2 = tplFaceRect[:4]

		# Clamp coordinates to image bounds
		iX1 = max(0, min(int(iX1), iW - 1))
		iY1 = max(0, min(int(iY1), iH - 1))
		iX2 = max(0, min(int(iX2), iW - 1))
		iY2 = max(0, min(int(iY2), iH - 1))

		# Validate rectangle
		if iX2 <= iX1 or iY2 <= iY1:
			logger.warning(f"Invalid face rect: ({iX1}, {iY1}, {iX2}, {iY2})")
			return image

		# Create image copy to avoid modifying original
		imgCopy = image.copy()

		# Draw bounding box
		cv2.rectangle(imgCopy, (iX1, iY1), (iX2, iY2), (0, 255, 0), 2)

		# Prepare text
		sText = f"{sEmotionLabel}: {fConfidence:.1f}%"
		iFont = cv2.FONT_HERSHEY_SIMPLEX
		fFontScale = 0.6
		iThickness = 2

		# Calculate text size
		(tplTextW, iTextH), iBaseline = cv2.getTextSize(
			sText, iFont, fFontScale, iThickness
		)

		# Calculate text position (above rectangle, or below if out of bounds)
		iTextX = iX1
		iTextY = iY1 - 10 if iY1 - 10 > 0 else iY1 + iTextH + 10

		# Calculate background rectangle bounds
		iBgYStart = max(0, iTextY - iTextH - iBaseline)
		iBgYEnd = min(iH, iTextY + iBaseline)
		iBgXStart = max(0, iTextX)
		iBgXEnd = min(iW, iTextX + tplTextW)

		# Draw background rectangle for text readability
		if iBgXStart < iBgXEnd and iBgYStart < iBgYEnd:
			cv2.rectangle(
				imgCopy,
				(iBgXStart, iBgYStart),
				(iBgXEnd, iBgYEnd),
				(0, 255, 0), -1
			)

		# Draw text
		cv2.putText(
			imgCopy, sText,
			(iTextX, iTextY),
			iFont, fFontScale,
			(0, 0, 0), iThickness
		)

		return imgCopy

	except Exception as e:
		logger.error(f"Safe draw failed: {str(e)}")
		import traceback
		logger.error(traceback.format_exc())
		return image