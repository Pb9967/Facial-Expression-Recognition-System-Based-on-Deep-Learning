# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆

"""
数据集类与加载器
实现情绪识别数据集的加载和处理
"""

import os
import torch
from PIL import Image
import numpy


# EmotionDataset
class EmotionDataset(torch.utils.data.Dataset):
	"""
	人脸情绪识别数据集类
	支持FER2013、CK+和自采图片组成的混合数据集
	"""

	def __init__(self, sDataDir, sSplit='train', _transform=None, tplTargetSize=(224, 224)):
		"""
		初始化数据集

		参数:
			sDataDir (str): 数据集根目录
			sSplit (str): 数据集划分，'train' 或 'val' 或 'test'
			_transform (callable): 数据增强变换
			tplTargetSize (tuple): 输出图像尺寸
		"""
		import config
		import transforms
		self.sDataDir = sDataDir
		self.sSplit = sSplit
		self._transform = _transform if _transform else transforms.get_transforms(sMode=sSplit)
		self.tplTargetSize = tplTargetSize
		self.lstImagePaths = []
		self.lstLabels = []

		# 加载数据集
		self._load_dataset()

	def _load_dataset(self):
		"""
		加载数据集，遍历指定划分的情绪类别文件夹
		"""
		import config
		import utils
		# 确定数据集划分的路径
		sSplitDir = os.path.join(self.sDataDir, self.sSplit)

		utils.logger.info(f"加载 {self.sSplit} 数据集，路径: {sSplitDir}")

		# 遍历每个情绪类别文件夹
		for sEmotion, iLabel in config.EMOTION_LABELS.items():
			sEmotionDir = os.path.join(sSplitDir, sEmotion)

			if not os.path.exists(sEmotionDir):
				utils.logger.warning(f"情绪类别文件夹 '{sEmotionDir}' 不存在，跳过该类别")
				continue

			# 遍历文件夹中的所有图像文件
			for sFilename in os.listdir(sEmotionDir):
				if sFilename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
					sImagePath = os.path.join(sEmotionDir, sFilename)

					# 检查图像有效性
					if utils.check_image_validity(sImagePath):
						self.lstImagePaths.append(sImagePath)
						self.lstLabels.append(iLabel)

		utils.logger.info(f"成功加载 {self.sSplit} 数据集: {len(self.lstImagePaths)} 张图像，包含 {len(config.EMOTION_LABELS)} 种情绪类别")

	def __len__(self):
		"""
		返回数据集大小
		"""
		return len(self.lstImagePaths)

	def __getitem__(self, iIdx):
		"""
		获取指定索引的图像和标签

		参数:
			iIdx (int): 索引

		返回:
			image (tensor): 预处理后的图像张量
			label (int): 情绪标签
		"""
		import config
		try:
			# 读取图像
			sImagePath = self.lstImagePaths[iIdx]
			imgImage = Image.open(sImagePath).convert('RGB')

			# 图像预处理
			if self._transform:
				imgImage = self._transform(imgImage)

			# 获取标签
			iLabel = self.lstLabels[iIdx]

			return imgImage, iLabel

		except Exception as e:
			import utils
			utils.logger.error(f"读取图像时出错: {self.lstImagePaths[iIdx]}, 错误信息: {str(e)}")
			# 出错时返回随机图像和标签
			arrDummy = numpy.random.randint(0, 255, (self.tplTargetSize[0], self.tplTargetSize[1], 3), dtype=numpy.uint8)
			imgDummyImage = Image.fromarray(arrDummy)
			if self._transform:
				imgDummyImage = self._transform(imgDummyImage)
			return imgDummyImage, numpy.random.randint(0, len(config.EMOTION_LABELS))


# get_data_loaders
def get_data_loaders(sDataDir, _fValSize=0.2, iBatchSize=32, iNumWorkers=4):
	"""
	获取训练、验证和测试数据加载器

	参数:
		sDataDir (str): 数据目录
		_fValSize (float): 验证集比例（相对于训练集）
		iBatchSize (int): 批次大小
		iNumWorkers (int): 工作进程数

	返回:
		train_loader, val_loader, test_loader: 数据加载器
	"""
	import utils
	objLogger = utils.setup_logger()

	objLogger.info(f"=== 准备数据加载器 ===")

	# 检查数据集结构
	for sSplit in ['train', 'val', 'test']:
		sSplitDir = os.path.join(sDataDir, sSplit)
		if not os.path.exists(sSplitDir):
			objLogger.error(f"{sSplit} 数据集目录不存在: {sSplitDir}")
			raise FileNotFoundError(f"{sSplit} 数据集目录不存在")

	# 创建训练集、验证集和测试集实例
	objTrainDataset = EmotionDataset(sDataDir, sSplit='train')
	objValDataset = EmotionDataset(sDataDir, sSplit='val')
	objTestDataset = EmotionDataset(sDataDir, sSplit='test')

	objLogger.info(f"训练集: {len(objTrainDataset)} 张图像")
	objLogger.info(f"验证集: {len(objValDataset)} 张图像")
	objLogger.info(f"测试集: {len(objTestDataset)} 张图像")

	# 检查数据集是否为空
	if len(objTrainDataset) == 0:
		objLogger.error("训练集为空，请检查数据文件")
		raise RuntimeError("训练集为空")

	if len(objValDataset) == 0:
		objLogger.error("验证集为空，请确保已创建验证集")
		raise RuntimeError("验证集为空")

	if len(objTestDataset) == 0:
		objLogger.error("测试集为空，请检查数据文件")
		raise RuntimeError("测试集为空")

	# 创建数据加载器
	objTrainLoader = torch.utils.data.DataLoader(
		objTrainDataset,
		batch_size=iBatchSize,
		shuffle=True,
		num_workers=iNumWorkers,
		pin_memory=True
	)

	objValLoader = torch.utils.data.DataLoader(
		objValDataset,
		batch_size=iBatchSize,
		shuffle=False,
		num_workers=iNumWorkers,
		pin_memory=True
	)

	objTestLoader = torch.utils.data.DataLoader(
		objTestDataset,
		batch_size=iBatchSize,
		shuffle=False,
		num_workers=iNumWorkers,
		pin_memory=True
	)

	objLogger.info(f"数据加载器创建完成")

	return objTrainLoader, objValLoader, objTestLoader