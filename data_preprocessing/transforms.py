# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆

"""
数据增强与预处理模块
包含Train/Val/Test差异化的变换策略
"""

import torchvision


# 模块级函数，用于替代lambda，确保可被pickle序列化（支持多进程DataLoader）
def _ensure_rgb(img):
	"""
	将非RGB模式的图像转换为RGB。
	用于处理FER2013中的灰度图（L模式），避免Normalize时通道不匹配。
	"""
	return img.convert('RGB') if img.mode != 'RGB' else img


# get_transforms
def get_transforms(sMode='train'):
	"""
	获取数据增强变换

	参数:
		sMode (str): 模式，'train' 或 'val' 或 'test'

	返回:
		transform (Compose): 数据增强变换组合
	"""
	import config

	# 基础变换：强制RGB、调整尺寸、转换为张量、归一化
	lstBaseTransforms = [
		torchvision.transforms.Lambda(_ensure_rgb),
		torchvision.transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)
	]

	if sMode == 'train':
		# 训练集数据增强：随机翻转、颜色抖动等
		lstTrainTransforms = [
			torchvision.transforms.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
			torchvision.transforms.RandomRotation(degrees=config.ROTATION_DEGREES),
			torchvision.transforms.ColorJitter(
				brightness=config.COLOR_JITTER_BRIGHTNESS,
				contrast=config.COLOR_JITTER_CONTRAST,
				saturation=config.COLOR_JITTER_SATURATION,
				hue=config.COLOR_JITTER_HUE
			),
			torchvision.transforms.RandomResizedCrop(config.IMAGE_HEIGHT, scale=config.RANDOM_CROP_SCALE),
		] + lstBaseTransforms

		return torchvision.transforms.Compose(lstTrainTransforms)

	elif sMode == 'val':
		# 验证集变换：只进行基础变换
		return torchvision.transforms.Compose(lstBaseTransforms)

	else:
		# 测试集变换：与验证集相同
		return torchvision.transforms.Compose(lstBaseTransforms)