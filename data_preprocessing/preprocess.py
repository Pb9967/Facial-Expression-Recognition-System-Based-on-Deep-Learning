# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆

"""
数据准备与预处理模块主程序
包含完整的数据处理流程
"""

import os
import shutil
import random
import sklearn


# create_dataset_structure
def create_dataset_structure(sOutputDir):
	"""
	创建数据集目录结构

	参数:
		sOutputDir (str): 输出目录

	返回:
		dict: 各子集路径
	"""
	import config
	import utils
	dctPaths = {
		'train': os.path.join(sOutputDir, 'train'),
		'val': os.path.join(sOutputDir, 'val'),
		'test': os.path.join(sOutputDir, 'test')
	}

	# 创建主目录
	for sKey, sValue in dctPaths.items():
		os.makedirs(sValue, exist_ok=True)

		# 创建每个情绪类别的子目录
		for sEmotion in config.EMOTION_LABELS.keys():
			sEmotionDir = os.path.join(sValue, sEmotion)
			os.makedirs(sEmotionDir, exist_ok=True)

	utils.logger.info("数据集目录结构创建成功")
	return dctPaths


# split_dataset
def split_dataset(sRawDataDir, dctOutputPaths):
	"""
	划分数据集

	参数:
		sRawDataDir (str): 原始数据目录
		dctOutputPaths (dict): 输出路径字典
	"""
	import config
	import utils
	utils.logger.info("开始划分数据集...")

	# 遍历每个情绪类别
	for sEmotion in config.EMOTION_LABELS.keys():
		sEmotionDir = os.path.join(sRawDataDir, sEmotion)

		if not os.path.exists(sEmotionDir):
			utils.logger.warning(f"情绪类别 '{sEmotion}' 目录不存在，跳过")
			continue

		# 获取该类别的所有图像路径
		lstImages = [f for f in os.listdir(sEmotionDir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

		# 检查图像有效性
		lstValidImages = []
		for sImg in lstImages:
			sImgPath = os.path.join(sEmotionDir, sImg)
			if utils.check_image_validity(sImgPath):
				lstValidImages.append(sImg)

		if not lstValidImages:
			utils.logger.warning(f"情绪类别 '{sEmotion}' 无有效图像，跳过")
			continue

		utils.logger.info(f"情绪类别 '{sEmotion}' 有 {len(lstValidImages)} 张有效图像")

		# 分层抽样划分
		# 第一步：划分训练集
		lstTrainImages, lstTempImages = sklearn.model_selection.train_test_split(
			lstValidImages,
			test_size=(config.VALIDATION_SPLIT + config.TEST_SPLIT),
			random_state=config.RANDOM_STATE,
			stratify=[sEmotion]*len(lstValidImages)
		)

		# 第二步：划分验证集和测试集
		fTestSize = config.TEST_SPLIT / (config.VALIDATION_SPLIT + config.TEST_SPLIT)
		lstValImages, lstTestImages = sklearn.model_selection.train_test_split(
			lstTempImages,
			test_size=fTestSize,
			random_state=config.RANDOM_STATE,
			stratify=[sEmotion]*len(lstTempImages)
		)

		utils.logger.info(f"情绪类别 '{sEmotion}' 划分: 训练集={len(lstTrainImages)}, 验证集={len(lstValImages)}, 测试集={len(lstTestImages)}")

		# 复制图像到相应目录
		for sImg in lstTrainImages:
			sSrcPath = os.path.join(sEmotionDir, sImg)
			sDestDir = os.path.join(dctOutputPaths['train'], sEmotion)
			utils.copy_image_to_dir(sSrcPath, sDestDir, sImg)

		for sImg in lstValImages:
			sSrcPath = os.path.join(sEmotionDir, sImg)
			sDestDir = os.path.join(dctOutputPaths['val'], sEmotion)
			utils.copy_image_to_dir(sSrcPath, sDestDir, sImg)

		for sImg in lstTestImages:
			sSrcPath = os.path.join(sEmotionDir, sImg)
			sDestDir = os.path.join(dctOutputPaths['test'], sEmotion)
			utils.copy_image_to_dir(sSrcPath, sDestDir, sImg)

	utils.logger.info("数据集划分完成")


# create_val_dataset_from_train
def create_val_dataset_from_train():
	"""
	从train目录中划分val验证集

	原始数据集结构已包含train和test目录
	需要从train目录中划分20%作为val验证集
	"""
	import config
	import utils
	objLogger = utils.setup_logger()

	objLogger.info("=== 创建val验证集 ===")

	# 检查原始数据集结构
	if not os.path.exists(config.DATASET_ROOT):
		objLogger.error(f"数据集根目录不存在: {config.DATASET_ROOT}")
		return False

	sTrainDir = os.path.join(config.DATASET_ROOT, 'train')
	sTestDir = os.path.join(config.DATASET_ROOT, 'test')
	sValDir = os.path.join(config.DATASET_ROOT, 'val')

	# 检查train和test目录是否存在
	if not os.path.exists(sTrainDir):
		objLogger.error(f"训练集目录不存在: {sTrainDir}")
		return False

	if not os.path.exists(sTestDir):
		objLogger.warning(f"测试集目录不存在: {sTestDir}")

	# 创建val目录结构
	for sEmotion in config.EMOTION_LABELS.keys():
		sEmotionValDir = os.path.join(sValDir, sEmotion)
		os.makedirs(sEmotionValDir, exist_ok=True)

	# 遍历每个情绪类别
	iTotalTrainImages = 0
	iTotalValImages = 0

	for sEmotion in config.EMOTION_LABELS.keys():
		sTrainEmotionDir = os.path.join(sTrainDir, sEmotion)

		if not os.path.exists(sTrainEmotionDir):
			objLogger.warning(f"训练集情绪类别 '{sEmotion}' 目录不存在: {sTrainEmotionDir}")
			continue

		# 获取该情绪类别的所有图像
		lstImages = [f for f in os.listdir(sTrainEmotionDir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

		if not lstImages:
			objLogger.warning(f"训练集情绪类别 '{sEmotion}' 无图像文件")
			continue

		# 验证图像有效性
		lstValidImages = []
		for sImgName in lstImages:
			sImgPath = os.path.join(sTrainEmotionDir, sImgName)
			if utils.check_image_validity(sImgPath):
				lstValidImages.append(sImgName)

		objLogger.info(f"情绪类别 '{sEmotion}' 有 {len(lstValidImages)} 张有效图像")

		# 使用分层抽样划分训练集和验证集
		# 验证集比例为20%
		iValSize = int(len(lstValidImages) * 0.2)
		iTrainSize = len(lstValidImages) - iValSize

		# 随机选择验证集图像
		random.seed(config.RANDOM_STATE)
		lstValImages = random.sample(lstValidImages, iValSize)

		# 复制验证集图像到val目录
		for sImgName in lstValImages:
			sSrcPath = os.path.join(sTrainEmotionDir, sImgName)
			sDestPath = os.path.join(sValDir, sEmotion, sImgName)
			shutil.copy(sSrcPath, sDestPath)

		objLogger.info(f"情绪类别 '{sEmotion}': 训练集={iTrainSize}, 验证集={iValSize}")

		iTotalTrainImages += iTrainSize
		iTotalValImages += iValSize

	objLogger.info(f"\n总统计: 训练集={iTotalTrainImages}, 验证集={iTotalValImages}")

	if iTotalValImages == 0:
		objLogger.error("未成功创建验证集")
		return False
	else:
		objLogger.info(f"验证集已成功创建在: {sValDir}")

	return True


# verify_dataset_structure
def verify_dataset_structure(sDataDir=None):
	"""
	验证数据集结构
	检查是否包含完整的train/val/test目录

	参数:
		sDataDir (str): 待验证的数据目录，默认为 config.PROCESSED_DATA_DIR
	"""
	import config
	import utils
	objLogger = utils.setup_logger()

	# 默认验证处理后数据目录
	if sDataDir is None:
		sDataDir = config.PROCESSED_DATA_DIR

	objLogger.info("=== 验证数据集结构 ===")

	lstRequiredDirs = ['train', 'val', 'test']

	for sSplit in lstRequiredDirs:
		sSplitDir = os.path.join(sDataDir, sSplit)

		if not os.path.exists(sSplitDir):
			objLogger.error(f"{sSplit} 目录不存在: {sSplitDir}")
			return False

		objLogger.info(f"{sSplit} 目录已创建: {sSplitDir}")

		# 检查每个情绪类别是否都有对应的文件夹
		for sEmotion in config.EMOTION_LABELS.keys():
			sEmotionDir = os.path.join(sSplitDir, sEmotion)

			if os.path.exists(sEmotionDir):
				iCount = len([f for f in os.listdir(sEmotionDir)
							 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

				if iCount > 0:
					objLogger.info(f"  {sEmotion}: {iCount} 张图像")
				else:
					objLogger.warning(f"  {sEmotion}: 无图像文件")
			else:
				objLogger.warning(f"  {sEmotion}: 目录不存在")

	objLogger.info("=== 数据集结构验证完成 ===")
	return True


# main
def main():
	"""
	直接运行的主程序函数
	不使用终端调用，直接使用内置参数
	"""
	import config
	import dataset
	import utils
	objLogger = utils.setup_logger()

	objLogger.info("=== 数据准备与预处理模块 ===")

	# 内置参数设置
	dctConfig = {
		"create_val": False,  # 当使用 RAW_DATA_DIR 划分时，不需要从train中再分val
		"verify": True,
		"load": True,
		"batch_size": 32,
		"num_workers": 4
	}

	objLogger.info("使用内置参数运行:")
	for sKey, _value in dctConfig.items():
		objLogger.info(f"  {sKey}: {_value}")

	# 确定输出目录
	sTargetDir = config.PROCESSED_DATA_DIR
	objLogger.info(f"处理后数据输出目录: {sTargetDir}")

	# 检查原始数据是否存在
	if not os.path.exists(config.RAW_DATA_DIR):
		objLogger.error(f"原始数据目录不存在: {config.RAW_DATA_DIR}")
		objLogger.info("请将未划分的原始情绪图像按类别文件夹存放至上述目录")
		objLogger.info("目录结构示例: raw/angry/*.jpg, raw/happy/*.jpg, ...")
		return False

	# 1. 创建标准化的数据集目录结构
	objLogger.info("=== 创建数据集目录结构 ===")
	dctPaths = create_dataset_structure(sTargetDir)

	# 2. 从原始数据目录划分到输出目录
	objLogger.info("=== 划分数据集 ===")
	split_dataset(config.RAW_DATA_DIR, dctPaths)
	objLogger.info("数据集划分完成")

	# 3. 验证输出目录的数据集结构
	if dctConfig["verify"]:
		objLogger.info("=== 验证数据集结构 ===")
		if verify_dataset_structure(sTargetDir):
			utils.print_emotion_distribution(sTargetDir)
		else:
			objLogger.error("数据集结构验证失败")
			return False

	# 4. 测试数据加载器
	if dctConfig["load"]:
		try:
			objLogger.info("=== 测试数据加载器 ===")
			objTrainLoader, objValLoader, objTestLoader = dataset.get_data_loaders(
				sDataDir=sTargetDir,
				iBatchSize=dctConfig["batch_size"],
				iNumWorkers=dctConfig["num_workers"]
			)

			objLogger.info(f"训练集加载器: {len(objTrainLoader)} 批次")
			objLogger.info(f"验证集加载器: {len(objValLoader)} 批次")
			objLogger.info(f"测试集加载器: {len(objTestLoader)} 批次")

			# 打印批次信息
			for i, (lstImages, lstLabels) in enumerate(objTrainLoader):
				objLogger.info(f"训练集批次 {i}: 图像形状 {lstImages.shape}, 标签形状 {lstLabels.shape}")
				objLogger.info(f"标签值: {lstLabels}")
				if i == 1:
					break

			objLogger.info("数据加载器测试成功")

		except Exception as e:
			objLogger.error(f"数据加载器测试失败: {str(e)}")
			return False

	objLogger.info("=== 所有处理完成 ===")
	return True


if __name__ == "__main__":
	# 直接运行主程序
	bSuccess = main()
	if bSuccess:
		print("程序执行成功！")
	else:
		print("程序执行失败！")