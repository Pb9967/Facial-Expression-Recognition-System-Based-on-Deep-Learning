# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆

"""
数据准备与预处理模块工具函数
"""

import os
import logging
import numpy
from PIL import Image
import shutil


# setup_logger
def setup_logger():
	"""
	设置日志系统
	"""
	import config
	objLogger = logging.getLogger(__name__)
	objLogger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))

	# 避免重复添加处理器导致日志重复输出
	if not objLogger.handlers:
		# 控制台处理器
		objConsoleHandler = logging.StreamHandler()
		objConsoleHandler.setLevel(logging.INFO)

		# 文件处理器
		objFileHandler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
		objFileHandler.setLevel(getattr(logging, config.LOG_LEVEL.upper()))

		# 格式化输出
		objFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		objConsoleHandler.setFormatter(objFormatter)
		objFileHandler.setFormatter(objFormatter)

		objLogger.addHandler(objConsoleHandler)
		objLogger.addHandler(objFileHandler)

	return objLogger


# 初始化日志器
logger = setup_logger()


# check_image_validity
def check_image_validity(sImagePath):
	"""
	检查图像文件的有效性

	参数:
		sImagePath (str): 图像文件路径

	返回:
		bool: 图像是否有效
	"""
	try:
		# 尝试打开图像文件
		with Image.open(sImagePath) as imgImg:
			imgImg.verify()

		# 检查图像尺寸
		with Image.open(sImagePath) as imgImg:
			tplSize = imgImg.size
			if tplSize[0] < 10 or tplSize[1] < 10:
				logger.warning(f"图像尺寸过小: {sImagePath}")
				return False

			# 检查图像内容是否有效
			arrImgArray = numpy.array(imgImg)
			if numpy.all(arrImgArray == 0) or numpy.all(arrImgArray == 255):
				logger.warning(f"图像内容无效: {sImagePath}")
				return False

		return True

	except Exception as e:
		logger.error(f"图像文件无效: {sImagePath}, 错误信息: {str(e)}")
		return False


# count_emotion_images
def count_emotion_images(sDataDir, sSplit='train'):
	"""
	统计每个情绪类别的图像数量

	参数:
		sDataDir (str): 数据目录
		sSplit (str): 数据集划分

	返回:
		dict: 每个情绪类别的图像数量
	"""
	import config
	dctEmotionCounts = {sEmotion: 0 for sEmotion in config.EMOTION_LABELS.keys()}

	sSplitDir = os.path.join(sDataDir, sSplit)

	for sEmotion in config.EMOTION_LABELS.keys():
		sEmotionDir = os.path.join(sSplitDir, sEmotion)

		if os.path.exists(sEmotionDir):
			iCount = len([
				f for f in os.listdir(sEmotionDir)
				if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
			])
			dctEmotionCounts[sEmotion] = iCount

	return dctEmotionCounts


# print_emotion_distribution
def print_emotion_distribution(sDataDir):
	"""
	打印情绪类别分布

	参数:
		sDataDir (str): 数据目录
	"""
	logger.info("=== 训练集情绪分布 ===")
	dctTrainCounts = count_emotion_images(sDataDir, 'train')
	for sEmotion, iCount in dctTrainCounts.items():
		logger.info(f"{sEmotion}: {iCount}")

	logger.info("\n=== 验证集情绪分布 ===")
	dctValCounts = count_emotion_images(sDataDir, 'val')
	for sEmotion, iCount in dctValCounts.items():
		logger.info(f"{sEmotion}: {iCount}")

	logger.info("\n=== 测试集情绪分布 ===")
	dctTestCounts = count_emotion_images(sDataDir, 'test')
	for sEmotion, iCount in dctTestCounts.items():
		logger.info(f"{sEmotion}: {iCount}")

	logger.info(f"\n总训练图像数: {sum(dctTrainCounts.values())}")
	logger.info(f"总验证图像数: {sum(dctValCounts.values())}")
	logger.info(f"总测试图像数: {sum(dctTestCounts.values())}")


# copy_image_to_dir
def copy_image_to_dir(sSrcPath, sDestDir, sFilename):
	"""
	复制图像到指定目录

	参数:
		sSrcPath (str): 源图像路径
		sDestDir (str): 目标目录
		sFilename (str): 文件名
	"""
	os.makedirs(sDestDir, exist_ok=True)
	sDestPath = os.path.join(sDestDir, sFilename)
	shutil.copy(sSrcPath, sDestPath)


# get_image_paths_by_emotion
def get_image_paths_by_emotion(sDataDir, sEmotion, sSplit='train'):
	"""
	获取指定情绪类别的所有图像路径

	参数:
		sDataDir (str): 数据目录
		sEmotion (str): 情绪类别
		sSplit (str): 数据集划分

	返回:
		list: 图像路径列表
	"""
	sSplitDir = os.path.join(sDataDir, sSplit)
	sEmotionDir = os.path.join(sSplitDir, sEmotion)

	if not os.path.exists(sEmotionDir):
		return []

	lstImagePaths = [
		os.path.join(sEmotionDir, sFilename)
		for sFilename in os.listdir(sEmotionDir)
		if sFilename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
	]

	return lstImagePaths