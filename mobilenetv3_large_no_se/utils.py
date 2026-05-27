# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆
"""
Utility functions for emotion recognition model.
Provides helper functions for training, inference and evaluation.
"""

import os
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import dlib


class ModelHelper:
    """
    Model helper class providing static utility methods.
    Handles weights loading, saving, and preprocessing.
    """

    @staticmethod
    def SetupLogger():
        """
        Setup logging system.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            sFormat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            formatter = logging.Formatter(sFormat)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @staticmethod
    def LoadWeights(model, sWeightPath):
        """
        Load model weights from file.

        Args:
            model (nn.Module): Model to load weights into.
            sWeightPath (str): Path to weight file.

        Returns:
            nn.Module: Model with loaded weights.

        Raises:
            FileNotFoundError: If weight file not found.
            RuntimeError: If loading fails.
        """
        if not os.path.exists(sWeightPath):
            raise FileNotFoundError(f"Weight file not found: {sWeightPath}")

        try:
            # Get device from model
            sDevice = next(model.parameters()).device
            model.load_state_dict(
                torch.load(sWeightPath, map_location=sDevice)
            )
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load weights: {str(e)}")

    @staticmethod
    def SaveWeights(model, sWeightPath):
        """
        Save model weights to file.

        Args:
            model (nn.Module): Model to save.
            sWeightPath (str): Path to save weight file.

        Raises:
            RuntimeError: If saving fails.
        """
        sDir = os.path.dirname(sWeightPath)
        if sDir:
            os.makedirs(sDir, exist_ok=True)

        try:
            torch.save(model.state_dict(), sWeightPath)
        except Exception as e:
            raise RuntimeError(f"Failed to save weights: {str(e)}")

    @staticmethod
    def PreprocessImage(image):
        """
        Preprocess image for model input.

        Args:
            image (np.ndarray): Input image (BGR format).

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        # Handle different image formats
        if image is None or image.size == 0:
            raise ValueError("Invalid image input")

        # Convert to RGB
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Define transform pipeline
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Apply transform and add batch dimension
        tensor = transform(image).unsqueeze(0)

        return tensor

    @staticmethod
    def PostprocessPredictions(predictions, lstLabels):
        """
        Convert model output to emotion label and confidence.

        Args:
            predictions (torch.Tensor): Model output tensor.
            lstLabels (list): List of emotion labels.

        Returns:
            tuple: (emotion_label, confidence)
        """
        # Apply softmax to get probabilities
        fProbs = torch.softmax(predictions, dim=0)

        # Get predicted class
        iEmotionIdx = torch.argmax(fProbs).item()
        sEmotionLabel = lstLabels[iEmotionIdx]
        fConfidence = fProbs[iEmotionIdx].item() * 100

        return sEmotionLabel, fConfidence

    @staticmethod
    def ComputeAccuracy(predictions, labels):
        """
        Compute classification accuracy.

        Args:
            predictions (torch.Tensor): Model predictions.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            float: Accuracy score.
        """
        _, predicted = torch.max(predictions, 1)
        iTotal = labels.size(0)
        iCorrect = (predicted == labels).sum().item()

        return iCorrect / iTotal

    @staticmethod
    def ComputeClassificationReport(predictions, labels, lstClassNames):
        """
        Generate classification report.

        Args:
            predictions (torch.Tensor): Model predictions.
            labels (torch.Tensor): Ground truth labels.
            lstClassNames (list): List of class names.

        Returns:
            str: Classification report string.
        """
        from sklearn.metrics import classification_report

        _, predicted = torch.max(predictions, 1)
        sReport = classification_report(
            labels.cpu().numpy(),
            predicted.cpu().numpy(),
            target_names=lstClassNames,
            output_dict=False
        )

        return sReport


class VisualizationHelper:
    """
    Visualization helper class for drawing results.
    """

    @staticmethod
    def DrawEmotionResult(image, sEmotionLabel, fConfidence, tplFaceRect):
        """
        Draw emotion recognition result on image.

        Args:
            image (np.ndarray): Input image (BGR format).
            sEmotionLabel (str): Predicted emotion label.
            fConfidence (float): Confidence score.
            tplFaceRect: Face bounding box (list/tuple or dlib.rectangle).

        Returns:
            np.ndarray: Image with drawn results.
        """
        # Parse face rectangle
        if isinstance(tplFaceRect, (list, tuple)):
            if len(tplFaceRect) >= 4:
                iX1, iY1, iX2, iY2 = tplFaceRect[:4]
            else:
                raise ValueError(
                    f"Invalid face rect, expected 4+ values, got {len(tplFaceRect)}"
                )
        elif isinstance(tplFaceRect, dlib.rectangle):
            iX1 = tplFaceRect.left()
            iY1 = tplFaceRect.top()
            iX2 = tplFaceRect.right()
            iY2 = tplFaceRect.bottom()
        else:
            raise TypeError(f"Unsupported face rect type: {type(tplFaceRect)}")

        # Draw bounding box
        cv2.rectangle(image, (iX1, iY1), (iX2, iY2), (0, 255, 0), 2)

        # Prepare text
        sText = f"{sEmotionLabel}: {fConfidence:.1f}%"
        tplTextSize, _ = cv2.getTextSize(
            sText, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        )

        # Calculate text position
        iTextX = iX1
        iTextY = iY1 - 10 if iY1 - 10 > 0 else iY1 + 30

        # Draw text background
        cv2.rectangle(
            image,
            (iTextX, iTextY - tplTextSize[1] - 10),
            (iTextX + tplTextSize[0] + 10, iTextY + 10),
            (0, 255, 0), -1
        )

        # Draw text
        cv2.putText(
            image, sText, (iTextX + 5, iTextY),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2
        )

        return image

    @staticmethod
    def VisualizePredictions(image, lstEmotions, lstConfidences, lstFaceRects):
        """
        Visualize emotion recognition results for multiple faces.

        Args:
            image (np.ndarray): Input image.
            lstEmotions (list): List of emotion labels.
            lstConfidences (list): List of confidence scores.
            lstFaceRects (list): List of face rectangles.

        Returns:
            np.ndarray: Image with visualized results.
        """
        resultImage = image.copy()

        for sEmotion, fConf, rect in zip(
            lstEmotions, lstConfidences, lstFaceRects
        ):
            resultImage = VisualizationHelper.DrawEmotionResult(
                resultImage, sEmotion, fConf, rect
            )

        return resultImage
