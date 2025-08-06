# ResNet-18 CIFAR-10 神经网络量化项目 / ResNet-18 CIFAR-10 Neural Network Quantization Project

本项目展示了在 CIFAR-10 数据集上训练 ResNet-18 模型的完整工作流程，包括模型训练、动态量化、静态量化（Eager Mode 和 FX Graph Mode）以及模型部署推理。代码采用模块化设计，便于理解和扩展。

This project demonstrates a complete workflow for training a ResNet-18 model on the CIFAR-10 dataset, including model training, dynamic quantization, static quantization (both Eager Mode and FX Graph Mode), and model deployment for inference. The code is structured modularly for clarity and extensibility.

## 完成的任务 / Completed Tasks

本项目实现了深度学习模型量化的完整流程：

This project implements a complete deep learning model quantization pipeline:

- **模型训练** / **Model Training**: 在 CIFAR-10 数据集上训练 ResNet-18 分类模型 / Train ResNet-18 classification model on CIFAR-10 dataset
- **动态量化** / **Dynamic Quantization**: 训练后量化，无需校准数据 / Post-training quantization without calibration data
- **静态量化** / **Static Quantization**: 使用校准数据的训练后量化，支持 Eager Mode 和 FX Graph Mode / Post-training quantization with calibration data, supporting both Eager Mode and FX Graph Mode
- **模型部署** / **Model Deployment**: 量化模型的推理部署 / Inference deployment of quantized models
- **性能评估** / **Performance Evaluation**: 准确率和模型大小的对比分析 / Comparative analysis of accuracy and model size

## 使用的模型和技术 / Models and Technologies Used

**模型架构** / **Model Architecture**:

- ResNet-18: 经典的残差神经网络，适用于图像分类任务 / Classic residual neural network for image classification tasks
- QuantizableResNet-18: 添加了量化桩(QuantStub/DeQuantStub)和量化友好操作的可量化版本 / Quantizable version with quantization stubs and quantization-friendly operations

**量化技术** / **Quantization Techniques**:

- **动态量化** / **Dynamic Quantization**: 运行时量化，主要针对线性层和卷积层 / Runtime quantization, mainly for linear and convolutional layers
- **静态量化(Eager Mode)** / **Static Quantization (Eager Mode)**: 使用 PyTorch 原生量化 API / Using PyTorch native quantization APIs
- **静态量化(FX Graph Mode)** / **Static Quantization (FX Graph Mode)**: 基于 FX 图模式的量化 / FX graph-based quantization

## 项目结构 / Project Structure

```
.
├── environment.yml           # Conda环境配置文件 / Conda environment definition
├── resnet.py                # 标准ResNet-18模型定义 / Standard ResNet-18 model definition
├── quantizable_resnet.py    # 可量化ResNet-18模型定义 / Quantizable ResNet-18 model definition
├── train.py                 # 模型训练脚本 / Model training script
├── quantize.py              # 量化脚本(动态+静态) / Quantization script (dynamic + static)
├── evaluate.py              # 模型评估脚本 / Model evaluation script
├── deploy.py                # 模型部署推理脚本 / Model deployment inference script
├── dataloader.py            # 数据加载和校准工具 / Data loading and calibration utilities
├── utils.py                 # 工具函数(模型保存/加载/比较) / Utility functions (model save/load/compare)
├── data/                    # 数据目录 / Data directory
├── weights/                 # 模型权重保存目录 / Model weights storage directory
└── README.md                # 项目说明文档 / Project documentation
```

## 文件功能说明 / File Functions

### 核心模型文件 / Core Model Files

- **`resnet.py`**: 实现标准的 ResNet-18 模型架构，包含 BasicBlock 和 ResNet 类定义，适用于常规训练和动态量化。
  / Implements standard ResNet-18 model architecture with BasicBlock and ResNet class definitions, suitable for regular training and dynamic quantization.

- **`quantizable_resnet.py`**: 实现量化友好的 ResNet-18 模型，添加了 QuantStub、DeQuantStub 和 FloatFunctional，支持静态量化。
  / Implements quantization-friendly ResNet-18 model with QuantStub, DeQuantStub, and FloatFunctional for static quantization support.

### 训练和数据处理 / Training and Data Processing

- **`train.py`**: 模型训练主脚本，实现 SGD 优化器、余弦退火学习率调度和完整的训练循环。
  / Main training script implementing SGD optimizer, cosine annealing scheduler, and complete training loop.

- **`dataloader.py`**: 数据加载工具，提供 CIFAR-10 数据加载器、校准数据创建和模型校准函数。
  / Data loading utilities providing CIFAR-10 data loaders, calibration data creation, and model calibration functions.

### 量化实现 / Quantization Implementation

- **`quantize.py`**: 量化实现脚本，包含动态量化和静态量化(Eager/FX 两种模式)的完整实现。
  / Quantization implementation script containing complete implementations of dynamic and static quantization (both Eager and FX modes).

### 评估和部署 / Evaluation and Deployment

- **`evaluate.py`**: 模型性能评估脚本，计算模型在测试集上的准确率。
  / Model performance evaluation script calculating accuracy on test dataset.

- **`deploy.py`**: 模型部署脚本，演示如何加载量化模型并进行单图片推理。
  / Model deployment script demonstrating how to load quantized models and perform single image inference.

- **`utils.py`**: 工具函数集合，包含模型保存/加载、性能计时装饰器和模型大小比较功能。
  / Utility functions collection including model save/load, performance timing decorator, and model size comparison features.

## 运行指南 / Running Guide

### 1. 环境配置 / Environment Setup

首先创建并激活 Conda 环境，安装所有必要的依赖包：

First, create and activate the Conda environment with all necessary dependencies:

```bash
conda env create -f environment.yml
conda activate torch-quan
```

环境包含的主要包 / Main packages included:

- PyTorch (CUDA 11.8 支持) / PyTorch (with CUDA 11.8 support)
- TorchVision (计算机视觉工具) / TorchVision (computer vision tools)
- ONNX (模型转换支持) / ONNX (model conversion support)
- TorchServe (模型服务部署) / TorchServe (model serving deployment)

### 2. 模型训练 / Model Training

训练 ResNet-18 模型在 CIFAR-10 数据集上，数据集将自动下载：

Train ResNet-18 model on CIFAR-10 dataset (dataset will be downloaded automatically):

```bash
python train.py
```

**训练配置** / **Training Configuration**:

- 训练轮数：10 epochs (演示用，可在 train.py 中调整) / Epochs: 10 (for demonstration, adjustable in train.py)
- 优化器：SGD (lr=0.01, momentum=0.9, weight_decay=5e-4) / Optimizer: SGD (lr=0.01, momentum=0.9, weight_decay=5e-4)
- 学习率调度：余弦退火 / LR Schedule: Cosine Annealing
- 输出：`./weights/resnet18_cifar10.pth` (权重) 和 `./weights/full_resnet18_cifar10.pth` (完整模型)
  / Output: `./weights/resnet18_cifar10.pth` (weights) and `./weights/full_resnet18_cifar10.pth` (full model)

### 3. 模型量化 / Model Quantization

执行动态量化和静态量化，自动生成校准数据：

Perform dynamic and static quantization with automatic calibration data generation:

```bash
python quantize.py
```

**量化过程** / **Quantization Process**:

1. **动态量化** / **Dynamic Quantization**: 对 Linear 和 Conv2d 层进行 int8 量化 / int8 quantization for Linear and Conv2d layers
2. **静态量化(Eager Mode)** / **Static Quantization (Eager Mode)**: 使用校准数据确定量化参数 / Use calibration data to determine quantization parameters
3. **静态量化(FX Graph Mode)** / **Static Quantization (FX Graph Mode)**: 基于计算图的量化优化 / Graph-based quantization optimization

**输出模型** / **Output Models**:

- `resnet18_cifar10_dynamic_quantized.pth`
- `resnet18_cifar10_static_quantized_eager.pth`
- `resnet18_cifar10_static_quantized_fx.pth`

### 4. 性能评估 / Performance Evaluation

比较原始模型和量化模型的准确率及文件大小：

Compare accuracy and file size between original and quantized models:

```bash
python evaluate.py
```

**评估指标** / **Evaluation Metrics**:

- 测试集准确率 / Test accuracy
- 模型文件大小 / Model file size
- 压缩比 / Compression ratio
- 推理时间 / Inference time

### 5. 模型部署 / Model Deployment

使用量化模型进行推理演示：

Demonstrate inference with quantized models:

```bash
python deploy.py
```

**部署特性** / **Deployment Features**:

- 自动加载动态量化模型 / Automatically load dynamic quantized model
- 支持单图片推理 / Support single image inference
- CPU 优化推理 / CPU-optimized inference
- CIFAR-10 类别预测 / CIFAR-10 class prediction

**注意** / **Note**: 脚本会自动创建示例图片进行演示，也可以提供真实图片路径。
/ The script automatically creates sample images for demonstration, or you can provide real image paths.

## 量化效果 / Quantization Results

**典型性能表现** / **Typical Performance**:

- 模型大小压缩：约 4 倍 / Model size reduction: ~4x
- 准确率损失：< 1% / Accuracy loss: < 1%
- CPU 推理速度：显著提升 / CPU inference speed: Significant improvement

## 扩展功能 / Extensions

### 量化感知训练 / Quantization-Aware Training (QAT)

可基于现有代码实现 QAT，通过在训练过程中模拟量化操作来进一步提升量化模型性能。

QAT can be implemented based on existing code to further improve quantized model performance by simulating quantization operations during training.

### ONNX 导出 / ONNX Export

支持将量化模型导出为 ONNX 格式，便于跨平台部署。

Support exporting quantized models to ONNX format for cross-platform deployment.

### 模型服务 / Model Serving

使用 TorchServe 实现量化模型的生产级部署。

Use TorchServe for production-level deployment of quantized models.
