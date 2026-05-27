# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆

"""
数据准备与预处理模块测试脚本
用于验证模块功能的完整性和正确性，并可视化数据分布
"""

import os
import sys
import pathlib
from PIL import Image


# 添加项目路径到系统路径
sys.path.append(str(pathlib.Path(__file__).parent))


# test_image_validation
def test_image_validation():
	"""
	测试图像验证功能
	"""
	import config
	import utils
	objLogger = utils.setup_logger()

	# 找到一张有效图像进行测试
	sSampleDir = os.path.join(config.DATASET_ROOT, "train", "happy")
	if os.path.exists(sSampleDir):
		lstImages = [f for f in os.listdir(sSampleDir) if f.lower().endswith('.jpg')]
		if lstImages:
			sTestImage = os.path.join(sSampleDir, lstImages[0])

			objLogger.info(f"测试图像: {sTestImage}")
			bIsValid = utils.check_image_validity(sTestImage)
			objLogger.info(f"图像有效性: {bIsValid}")

			if bIsValid:
				imgImg = Image.open(sTestImage)
				objLogger.info(f"图像尺寸: {imgImg.size}")
				objLogger.info(f"图像模式: {imgImg.mode}")
				objLogger.info("图像验证测试通过")
			else:
				objLogger.error("图像验证测试失败")
		else:
			objLogger.warning("未找到测试图像")
	else:
		objLogger.warning("测试目录不存在")


# test_data_loader
def test_data_loader():
	"""
	测试数据加载器功能
	"""
	import config
	import dataset
	import utils
	objLogger = utils.setup_logger()

	objLogger.info("=== 测试数据加载器 ===")

	try:
		objTrainLoader, objValLoader, objTestLoader = dataset.get_data_loaders(
			sDataDir=config.DATASET_ROOT,
			iBatchSize=16,
			iNumWorkers=2
		)

		objLogger.info("数据加载器创建成功")

		# 测试数据加载器的基本功能
		objLogger.info(f"训练集大小: {len(objTrainLoader.dataset)}")
		objLogger.info(f"验证集大小: {len(objValLoader.dataset)}")
		objLogger.info(f"测试集大小: {len(objTestLoader.dataset)}")

		# 测试批次数据
		for lstImages, lstLabels in objTrainLoader:
			objLogger.info(f"图像张量形状: {lstImages.shape}")
			objLogger.info(f"标签张量形状: {lstLabels.shape}")
			objLogger.info(f"批次大小: {lstImages.shape[0]}")
			objLogger.info(f"图像通道: {lstImages.shape[1]}")
			objLogger.info(f"图像尺寸: {lstImages.shape[2]}x{lstImages.shape[3]}")
			break

		objLogger.info("数据加载器测试通过")

	except Exception as e:
		objLogger.error(f"数据加载器测试失败: {str(e)}")


# test_transforms
def test_transforms():
	"""
	测试数据增强变换
	"""
	import config
	import transforms
	import utils
	objLogger = utils.setup_logger()

	objLogger.info("=== 测试数据增强变换 ===")

	# 获取训练集和测试集的变换
	objTrainTransform = transforms.get_transforms(sMode='train')
	objTestTransform = transforms.get_transforms(sMode='test')

	objLogger.info("变换定义成功")

	# 找到一张测试图像
	sSampleDir = os.path.join(config.DATASET_ROOT, "train", "happy")
	if os.path.exists(sSampleDir):
		lstImages = [f for f in os.listdir(sSampleDir) if f.lower().endswith('.jpg')]
		if lstImages:
			sTestImage = os.path.join(sSampleDir, lstImages[0])

			# 测试变换
			imgImg = Image.open(sTestImage)

			tensorTrainTransformed = objTrainTransform(imgImg)
			tensorTestTransformed = objTestTransform(imgImg)

			objLogger.info(f"训练集变换后形状: {tensorTrainTransformed.shape}")
			objLogger.info(f"测试集变换后形状: {tensorTestTransformed.shape}")

			# 检查图像范围
			objLogger.info(f"训练集变换后值范围: [{tensorTrainTransformed.min():.2f}, {tensorTrainTransformed.max():.2f}]")
			objLogger.info(f"测试集变换后值范围: [{tensorTestTransformed.min():.2f}, {tensorTestTransformed.max():.2f}]")

			objLogger.info("数据增强变换测试通过")
		else:
			objLogger.warning("未找到测试图像")
	else:
		objLogger.warning("测试目录不存在")


# test_dataset_distribution
def test_dataset_distribution():
	"""
	测试数据集分布
	"""
	import config
	import utils
	objLogger = utils.setup_logger()

	objLogger.info("=== 测试数据集分布 ===")

	dctTrainCounts = utils.count_emotion_images(config.DATASET_ROOT, sSplit='train')
	dctValCounts = utils.count_emotion_images(config.DATASET_ROOT, sSplit='val')
	dctTestCounts = utils.count_emotion_images(config.DATASET_ROOT, sSplit='test')

	objLogger.info("训练集分布:")
	for sEmotion, iCount in dctTrainCounts.items():
		objLogger.info(f"  {sEmotion}: {iCount}")

	objLogger.info("\n验证集分布:")
	for sEmotion, iCount in dctValCounts.items():
		objLogger.info(f"  {sEmotion}: {iCount}")

	objLogger.info("\n测试集分布:")
	for sEmotion, iCount in dctTestCounts.items():
		objLogger.info(f"  {sEmotion}: {iCount}")

	objLogger.info(f"\n训练集总数: {sum(dctTrainCounts.values())}")
	objLogger.info(f"验证集总数: {sum(dctValCounts.values())}")
	objLogger.info(f"测试集总数: {sum(dctTestCounts.values())}")

	# 计算类别平衡
	fTrainBalance = max(dctTrainCounts.values()) / min(dctTrainCounts.values())
	fValBalance = max(dctValCounts.values()) / min(dctValCounts.values())
	fTestBalance = max(dctTestCounts.values()) / min(dctTestCounts.values())

	objLogger.info(f"\n训练集类别平衡系数: {fTrainBalance:.2f}")
	objLogger.info(f"验证集类别平衡系数: {fValBalance:.2f}")
	objLogger.info(f"测试集类别平衡系数: {fTestBalance:.2f}")

	objLogger.info("数据集分布测试通过")


# SetDataShow
def SetDataShow():
	"""
	使用图表展示数据分布，并将生成的图片保存在 output 目录下。
	绘制训练集、验证集、测试集的情绪类别分布柱状图与饼图。
	"""
	import os
	import matplotlib
	matplotlib.use('Agg')  # 使用非交互式后端，避免无GUI环境报错
	import matplotlib.pyplot as plt
	import numpy as np
	import config
	import utils

	# 获取各数据集分布
	dctTrain = utils.count_emotion_images(config.DATASET_ROOT, 'train')
	dctVal = utils.count_emotion_images(config.DATASET_ROOT, 'val')
	dctTest = utils.count_emotion_images(config.DATASET_ROOT, 'test')

	lstLabels = list(config.EMOTION_LABELS.keys())
	arrTrain = np.array([dctTrain.get(k, 0) for k in lstLabels])
	arrVal = np.array([dctVal.get(k, 0) for k in lstLabels])
	arrTest = np.array([dctTest.get(k, 0) for k in lstLabels])

	# 设置中文字体与负号显示
	plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
	plt.rcParams['axes.unicode_minus'] = False

	# 创建画布
	fig = plt.figure(figsize=(16, 10))

	# 子图1：分组柱状图（train/val/test 对比）
	ax1 = fig.add_subplot(2, 2, 1)
	x = np.arange(len(lstLabels))
	width = 0.25
	ax1.bar(x - width, arrTrain, width, label='Train', color='#3498db', edgecolor='white')
	ax1.bar(x, arrVal, width, label='Val', color='#e67e22', edgecolor='white')
	ax1.bar(x + width, arrTest, width, label='Test', color='#2ecc71', edgecolor='white')
	ax1.set_xlabel('Emotion Category', fontsize=11)
	ax1.set_ylabel('Sample Count', fontsize=11)
	ax1.set_title('Dataset Distribution by Emotion (Bar Chart)', fontsize=13, fontweight='bold')
	ax1.set_xticks(x)
	ax1.set_xticklabels(lstLabels, rotation=30, ha='right')
	ax1.legend(loc='upper right')
	ax1.grid(axis='y', linestyle='--', alpha=0.6)

	# 子图2：训练集饼图
	ax2 = fig.add_subplot(2, 2, 2)
	colors = plt.cm.Set3(np.linspace(0, 1, len(lstLabels)))
	wedges, texts, autotexts = ax2.pie(
		arrTrain, labels=lstLabels, autopct='%1.1f%%',
		startangle=90, colors=colors, textprops={'fontsize': 9}
	)
	ax2.set_title('Training Set Distribution (Pie Chart)', fontsize=13, fontweight='bold')

	# 子图3：验证集饼图
	ax3 = fig.add_subplot(2, 2, 3)
	ax3.pie(
		arrVal, labels=lstLabels, autopct='%1.1f%%',
		startangle=90, colors=colors, textprops={'fontsize': 9}
	)
	ax3.set_title('Validation Set Distribution (Pie Chart)', fontsize=13, fontweight='bold')

	# 子图4：测试集饼图
	ax4 = fig.add_subplot(2, 2, 4)
	ax4.pie(
		arrTest, labels=lstLabels, autopct='%1.1f%%',
		startangle=90, colors=colors, textprops={'fontsize': 9}
	)
	ax4.set_title('Test Set Distribution (Pie Chart)', fontsize=13, fontweight='bold')

	plt.tight_layout(pad=3.0)

	# 保存图片至 output 目录
	sOutputDir = "output"
	os.makedirs(sOutputDir, exist_ok=True)
	sSavePath = os.path.join(sOutputDir, "emotion_distribution.png")
	plt.savefig(sSavePath, dpi=150, bbox_inches='tight')
	plt.close(fig)

	objLogger = utils.setup_logger()
	objLogger.info(f"数据分布图表已保存至: {os.path.abspath(sSavePath)}")


# run_all_tests
def run_all_tests():
	"""
	运行所有测试
	"""
	import utils
	objLogger = utils.setup_logger()

	objLogger.info("=== 运行所有测试 ===")

	lstTests = [
		test_image_validation,
		test_data_loader,
		test_transforms,
		test_dataset_distribution
	]

	iPassed = 0
	iFailed = 0

	for funcTest in lstTests:
		try:
			objLogger.info(f"\n--- 测试: {funcTest.__name__} ---")
			funcTest()
			iPassed += 1
		except Exception as e:
			objLogger.error(f"测试失败: {str(e)}")
			iFailed += 1

	objLogger.info(f"\n=== 测试结果 ===")
	objLogger.info(f"通过: {iPassed}")
	objLogger.info(f"失败: {iFailed}")

	if iFailed == 0:
		objLogger.info("所有测试通过！")
	else:
		objLogger.warning(f"有 {iFailed} 个测试失败")


if __name__ == "__main__":
	# 创建输出目录
	if not os.path.exists("output"):
		os.makedirs("output")

	# 运行全部测试
	run_all_tests()

	# 生成数据分布可视化图表
	SetDataShow()