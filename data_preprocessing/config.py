# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆

"""
数据准备与预处理模块配置文件
定义所有参数和路径设置
"""

import os


# 数据集路径
DATASET_ROOT = r"..\data_file\fer2013"
RAW_DATA_DIR = r"..\data_file\fer2013\raw"

# 输出路径
OUTPUT_DIR = r"..\outputs"
PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, "processed_data")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# 图像尺寸
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3

# 数据集划分比例
TRAIN_SPLIT = 0.6
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2

# 训练参数
BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_STATE = 42

# 数据增强参数
ROTATION_DEGREES = 10
HORIZONTAL_FLIP_PROB = 0.5
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2
COLOR_JITTER_HUE = 0.1
RANDOM_CROP_SCALE = (0.8, 1.0)

# 日志配置
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(LOG_DIR, "data_preprocessing.log")

# 情绪类别定义
EMOTION_LABELS = {
	'angry': 0,
	'disgust': 1,
	'fear': 2,
	'happy': 3,
	'sad': 4,
	'surprise': 5,
	'neutral': 6
}

EMOTION_NAMES = {v: k for k, v in EMOTION_LABELS.items()}

# 创建必要的目录（仅创建实际使用的目录）
for sDirectory in [OUTPUT_DIR, PROCESSED_DATA_DIR, LOG_DIR]:
	os.makedirs(sDirectory, exist_ok=True)