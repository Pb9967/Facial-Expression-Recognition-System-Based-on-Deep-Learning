# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Demo script for face detection and alignment module.
Shows how to use FaceDetector and FaceAligner for face processing.
"""

import os
import sys
import dlib
import logging
from pathlib import Path

# Add project root to system path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from face_detection_alignment.detector import FaceDetector
from face_detection_alignment.aligner import FaceAligner
from face_detection_alignment.utils import load_image, save_image


def main():
	"""Main function demonstrating face detection and alignment."""
	logger = logging.getLogger(__name__)

	# Initialize detector and aligner
	try:
		detector = FaceDetector()
		aligner = FaceAligner()
		logger.info("Detector and aligner initialized successfully")
	except Exception as e:
		logger.error(f"Initialization failed: {str(e)}")
		return

	# Test image path
	s_test_image = os.path.join(
		Path(__file__).parent,
		"..",
		"test_images",
		"face_2.png"
	)

	# Check if test image exists
	if not os.path.exists(s_test_image):
		logger.error(f"Test image not found: {s_test_image}")
		return

	# Create output directory
	s_output_dir = os.path.join(Path(__file__).parent, "outputs")
	os.makedirs(s_output_dir, exist_ok=True)

	try:
		# Load image
		logger.info(f"Loading image: {s_test_image}")
		image = load_image(s_test_image)

		# Face detection
		logger.info("Starting face detection")
		faces = detector.detect_faces(image)

		if not faces:
			logger.warning("No faces detected")
			return

		logger.info(f"Detected {len(faces)} face(s)")

		# Face alignment
		logger.info("Starting face alignment")
		aligned_faces = []

		for (x1, y1, x2, y2, f_confidence) in faces:
			logger.debug(f"Aligning face with confidence: {f_confidence:.2f}")

			try:
				# Detect landmarks for this face region
				landmarks = aligner.detect_landmarks(
					image,
					[dlib.rectangle(x1, y1, x2, y2)]
				)

				if landmarks:
					# Align face using detected landmarks
					aligned_face = aligner.align_face(image, landmarks[0])
					if aligned_face is not None:
						aligned_faces.append(aligned_face)

			except Exception as e:
				logger.error(f"Face alignment failed: {str(e)}")

		# Save aligned faces
		logger.info(f"Saving {len(aligned_faces)} aligned face(s)")
		for i, face in enumerate(aligned_faces):
			s_save_path = os.path.join(
				s_output_dir, f"aligned_face_{i + 1}.png"
			)
			save_image(face, s_save_path)
			logger.info(f"Saved to: {s_save_path}")

		# Visualize detection results
		logger.info("Drawing detection and alignment results")
		result_image = detector.draw_detection_results(image, faces)

		# Draw landmarks on result image
		all_landmarks = aligner.detect_landmarks(image)
		if all_landmarks:
			result_image = aligner.draw_landmarks(result_image, all_landmarks)

		save_image(
			result_image,
			os.path.join(s_output_dir, "detection_visualization.png")
		)

		logger.info("Processing complete")

	except Exception as e:
		import traceback
		logger.error(f"Processing failed: {str(e)}")
		logger.error(traceback.format_exc())


if __name__ == "__main__":
	main()