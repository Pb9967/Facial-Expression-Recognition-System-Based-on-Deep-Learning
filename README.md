# 基于深度学习的人脸表情识别系统

## 一、项目简介

本项目面向实时应用场景，设计并实现了一套轻量化人脸情绪识别系统。系统采用"人脸检测—关键点对齐—情绪分类"三级流水线架构：以OpenCV DNN完成人脸定位，Dlib 68点关键点驱动仿射对齐，MobileNetV3系列网络完成七类表情预测；前端基于PyQt5多线程架构实现交互式GUI，支持摄像头实时视频、单张图片和批量视频三种工作模式。为验证注意力机制在轻量级FER模型中的适用性，项目构造了Small/Large两种骨干与有/无SE模块的四组模型变体，开展了系统性的对比实验与消融分析。

## 二、环境配置

### 基础环境

- Python >= 3.10
- NVIDIA GPU（推荐，支持 CUDA）
- Windows

### Python 依赖

```bash
pip install -r requirements.txt
```

> **注意**：`torch` 和 `torchvision` 需根据你的 CUDA 版本选择对应版本。
> 例如 CUDA 12.x 用户应执行：
> ```bash
> pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu12x
> ```

### 环境验证

运行环境测试脚本：

```bash
python env_test.py
```

#### 本地测试运行输出结果示例

```
Python版本		3.12.12
CUDA可用:		True
CUDA版本:		12.8
GPU设备数:		1
当前GPU:		0
GPU名称:		NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

## 三、使用方法

### 1. 数据集准备

数据集准备
本项目数据处理模块支持两种数据源接入方式，分别对应不同的目录结构和处理流程。
#### 方式一：原始未划分数据
如果你拥有一批按情绪类别分类但尚未划分训练/验证/测试集的原始图像，按以下步骤处理。
* 第一步：准备原始数据目录结构

将原始图像按情绪类别分别存放至../data_file/fer2013/raw/ 目录下，结构如下：
```plaintext
data_file/fer2013/raw/
├── angry/
├── disgust/
├── fear/
├── happy/
├── sad/
├── surprise/
└── neutral/
```
每个子目录内存放对应情绪类别的原始图像文件，支持`.jpg`、`.jpeg`、`.png`格式。

* 第二步：运行预处理主程序
```python
cd data_preprocessing
python preprocess.py
```
preprocess.py 将自动执行以下操作：

在 outputs/processed_data/ 下创建标准的 train/val/test 三级目录结构，每个子集下包含7个情绪类别子目录；

从 raw/ 下的原始数据中按 6:2:2 的比例进行分层抽样划分，分层策略确保每个子集中各类别样本比例与总体保持一致；

划分过程中自动校验图像有效性（检查图像完整性、尺寸和内容有效性）；
划分完成后输出各类别在各子集中的分布统计；
自动创建 PyTorch DataLoader 并测试加载，验证整个数据链路通畅。
处理完成后，数据集结构如下：
```plaintext
outputs/processed_data/
├── train/          # 60% 训练集
├── val/            # 20% 验证集
└── test/           # 20% 测试集
```

#### 方式二：已含官方划分的数据集（如 FER2013）
如果你使用的数据集已包含官方划分的 train 和 test 目录，仅需从训练集中划分出验证集。
* 第一步：放置数据集
将数据集按以下结构放置：
```plaintext
data_file/fer2013/
├── train/          # 官方训练集
└── test/           # 官方测试集
```
* 第二步：从训练集中划分验证集
在 Python 交互环境或脚本中调用：
```python
from preprocess import create_val_dataset_from_train
create_val_dataset_from_train()
```
该函数将从 train/ 下的每个情绪类别中随机抽取 20% 的样本复制到 val/ 目录，剩余 80% 保留在训练集中。划分完成后结构如下：

```plaintext
data_file/fer2013/
├── train/
├── val/            # 从 train 中划分 20%
└── test/
```

#### 运行测试脚本验证整个数据处理链路：
```python
cd data_preprocessing
python test_preprocessing.py
```

该脚本将依次执行以下测试：

    图像有效性验证：随机抽取样本检查图像完整性；
    数据加载器测试：创建 DataLoader 并检查批次数据的形状和类型；
    变换策略测试：对比训练集与测试集变换后的张量形状和数值范围；
    分布统计测试：输出 train/val/test 各子集的情绪类别分布及类别平衡系数；
    可视化图表生成：在 output/ 目录下生成分组柱状图和各子集饼图，直观展示数据分布。
    所有测试通过且分布合理后，即可进入模型训练阶段。

### 2. 模型训练

项目包含四种模型变体，分别对应四个独立目录。进入目标模型目录后，运行训练脚本：

#### 主模型（MobileNetV3-Small + SE）

```bash
cd mobilenetv3_small_se
python train.py
```

训练过程中会自动保存最优权重和训练曲线。训练历史以JSON格式持久化存储，便于后续分析和可视化。

#### 其他模型变体

```bash
# Small 无 SE 消融基线
cd mobilenetv3_small_no_se
python train.py

# Large + SE 扩展模型
cd mobilenetv3_large_se
python train.py

# Large 无 SE 扩展消融基线
cd mobilenetv3_large_no_se
python train.py
```

所有模型共享统一的训练配置：120个epoch，Adam优化器，余弦退火学习率调度，交叉熵损失函数。训练结束时保存最终模型权重。

#### 模型推理测试

训练完成后，进入对应模型目录运行推理脚本：

```bash
cd mobilenetv3_small_se
python predict.py
```

### 3. GUI 系统使用

#### 启动 GUI

```bash
cd gui
python main_window.py
```

#### 操作流程

1. **选择模型**：启动后先在界面左侧选择一种模型（mobilenetv3_small_se / mobilenetv3_small_no_se / mobilenetv3_large_se / mobilenetv3_large_no_se），点击对应按钮加载权重。
2. **选择输入模式**：
   - **摄像头实时模式**：点击"打开摄像头"按钮，系统将调用默认摄像头设备，实时显示人脸检测框、情绪类别标签和置信度分数，右下角实时绘制情绪变化曲线。
   - **单张图片模式**：点击"选择图片"按钮，在弹出的文件对话框中选择待分析的图片（支持JPG/PNG/BMP格式），系统将逐张进行人脸检测与情绪识别，结果叠加显示在图像上。
   - **视频文件模式**：点击"选择视频"按钮，选择本地视频文件，系统逐帧处理并实时展示识别结果。处理完成后可导出完整的情绪变化时序数据。
3. **停止处理**：点击"停止"按钮结束当前识别任务，释放摄像头或视频文件资源。
4. **结果保存**：系统支持将识别结果导出为JSON/CSV格式，便于后续离线分析。

#### GUI 界面说明

界面采用左右分栏布局：
- **左侧**：图像显示区、模型选择按钮组、功能按钮行（打开摄像头/选择图片/选择视频/停止）
- **右侧**：情绪变化曲线图、识别历史记录列表
- **底部状态栏**：实时显示当前帧率、识别情绪类别、置信度、加载的模型名称

### 4. 模型评估

运行评估脚本生成混淆矩阵和各项性能指标：

```bash
cd evaluation
python evaluate.py
```

## 四、项目目录结构

```plaintext
codefile/						# 项目根目录
│
├── data_file/                        # 数据集存放目录
│   └── fer2013/
│       ├── train/                    # 训练集
│       ├── val/                      # 验证集
│       └── test/                     # 测试集
│
├── data_preprocessing/               # 数据预处理模块
│   ├── config.py                  # 数据集路径、划分比例等配置
│   ├── dataset.py                 # 自定义 Dataset 类，加载数据
│   ├── preprocess.py              # 完整数据预处理流程
│   ├── transforms.py              # 训练/验证/测试的数据增强策略
│   ├── utils.py                   # 辅助函数（计数、检查图像有效性等）
│   └── test_preprocessing.py      # 预处理模块测试脚本
│
├── mobilenetv3_small_se/               # 情绪识别模型mobilenetv3_small_se
│   ├── train_1p/                       # 示例：训练 1 轮生成的目录
│   │   ├── outputs/                    # 该次训练的输出结果
│   │   ├── weights/                    # 该次训练保存的模型权重
│   │   │   ├── 1p_best_model.pth       # 训练过程中验证准确率最高的模型
│   │   │   └── final_model.pth         # 训练结束后的最终模型
│   │   └── logs/                       # 该次训练产生的日志与曲线
│   │       ├── train_history_1p.json   # 训练历史记录（损失、准确率等）
│   │       └── training_curves_1p.png  # 训练曲线图（损失曲线和准确率曲线）
│   ├── __init__.py                     # 模块入口
│   ├── config.py                       # 模型配置
│   ├── model.py                        # 模型定义
│   ├── predict.py                      # 预测脚本
│   ├── train.py                        # 训练脚本
│   └── utils.py                        # 工具函数
│
├── mobilenetv3_small_no_se/            # 对比模型mobilenetv3_small_no_se
│   └── ...与mobilenetv3_small_se一致
│
├── mobilenetv3_large_se/               # 对比模型mobilenetv3_large_se
│   └── ...与mobilenetv3_small_se一致
│
├── mobilenetv3_large_no_se/            # 对比模型mobilenetv3_large_no_se
│   └── ...与mobilenetv3_small_se一致
│
├── evaluation/
│	├── outputs/				# 混淆矩阵保存目录
│	├── reports/				# 评估报告保存目录
│	├── __init__.py				# 模块入口
│	├── config.py				# 评估配置
│	├── utils.py				# 工具函数
│	├── metrics.py				# 指标计算
│	├── confusion_matrix.py		# 混淆矩阵可视化
│	├── speed.py				# 推理速度测试
│	├── robustness.py			# 鲁棒性分析
│	├── report.py				# 报告生成器
│	└── evaluate.py				# 评估入口脚本
│
├── face_detection_alignment/      # 人脸检测与对齐模块
│   ├── models/                    # 预训练模型文件
│   │   ├── deploy.prototxt        # OpenCV DNN 人脸检测模型描述文件
│   │   ├── res10_300x300_ssd_iter_140000.caffemodel
│   │   └── shape_predictor_68_face_landmarks.dat  # dlib 关键点检测器
│   ├── outputs/                   # 输出结果保存目录
│   ├── aligner.py                 # 人脸对齐类
│   ├── detector.py                # 人脸检测类
│   ├── for_use.py                 # 模块使用示例
│   └── utils.py                   # 辅助函数
│
├── gui/                           # 图形用户界面模块
│   ├── widgets/                   # 自定义界面组件
│   │   ├── camera_widget.py       # 摄像头线程封装
│   │   └── image_widget.py        # 图像显示组件
│   ├── main_window.py             # 主窗口程序
│   └── gui_utils.py               # GUI 专用安全绘图函数
│
├── test_images/                   # 测试图片（可选）
├── env_test.py                    # 环境测试代码
├── requirements.txt               # 环境配置说明文档
│
└── README.md                      # 项目说明文档（本文件）
```
