"""
quantize.py

实现动态量化和静态量化。
"""
import copy
import torch
import torchao.quantization.pt2e as pt2e
from torch.ao.quantization import get_default_qconfig, prepare, convert
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

from dataloader import create_calibration_loader, get_calibration_dataset, get_dataloader
from utils import save_model, compare_model_sizes, load_model, load_quantizable_model, timed
from evaluate import evaluate_model

def calibrate_model(model, data_loader, num_batches=10):
    """
    校准函数：在校准数据上运行模型以收集统计信息
    
    Args:
        model: 准备好的模型（包含观察器）
        data_loader: 校准数据加载器
        num_batches: 用于校准的批次数量
    """

    pt2e.move_exported_model_to_eval(model)
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            model(data)
            if batch_idx >= num_batches - 1:
                break

def dynamic_quantize_model(model):
    """
    对模型进行动态量化。
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,  # the model to be quantized
        {torch.nn.Linear, torch.nn.Conv2d},  # a set of layers to dynamically quantize
        dtype=torch.qint8)  # the target dtype for quantized weights

    return quantized_model

def static_quantize_model_eager(model, calibration_loader, backend='x86'):
    """
    使用静态量化对可量化模型进行量化
    
    Args:
        model: 可量化的模型 (QuantizableResNet)
        calibration_loader: 校准数据加载器
        backend: 量化后端 ('x86' 或 'qnnpack')
    
    Returns:
        quantized_model: 量化后的模型
    """
    # 0. 创建模型的深拷贝以避免修改原始模型
    model = copy.deepcopy(model)
    
    # 1. 设置为评估模式
    model.eval()
    
    # 2. 融合模块 (Conv-BN-ReLU)
    # 注意：这里需要根据实际模型结构调整融合模式
    modules_to_fuse = []
    # 添加主干网络的融合
    modules_to_fuse.append(['conv1', 'bn1'])
    
    # 为每个layer添加融合模式
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for i, block in enumerate(layer):
            block_name = f"{layer_name}.{i}"
            modules_to_fuse.append([f"{block_name}.conv1", f"{block_name}.bn1"])
            modules_to_fuse.append([f"{block_name}.conv2", f"{block_name}.bn2"])
            # 如果有shortcut，也要融合
            if len(block.shortcut) > 0:
                modules_to_fuse.append([f"{block_name}.shortcut.0", f"{block_name}.shortcut.1"])
    
    try:
        model_fused = torch.ao.quantization.fuse_modules(model, modules_to_fuse)
        print("✓ 模块融合完成")
    except Exception as e:
        print(f"警告：模块融合失败: {e}")
        print("继续使用原始模型...")
        model_fused = model
    
    # 3. 设置量化配置
    model_fused.qconfig = get_default_qconfig(backend)
    
    # 4. 准备模型（插入观察器）
    model_prepared = prepare(model_fused)
    print("模型准备完成，观察器已插入")
    
    # 5. 校准阶段
    print("开始校准...")
    calibrate_model(model_prepared, calibration_loader, num_batches=10)  # 明确指定批次数
    print("校准完成")
    
    # 6. 转换为量化模型
    model_quantized = convert(model_prepared)
    print("模型量化完成")
    
    return model_quantized

def static_quantize_model_fx(model, calibration_loader, backend='x86'):
    """
    使用FX图模式进行静态量化
    
    Args:
        model: 可量化的模型
        calibration_loader: 校准数据加载器
        backend: 量化后端
    
    Returns:
        quantized_model: 量化后的模型
    """
    # 0. 创建模型的深拷贝以避免修改原始模型
    model = copy.deepcopy(model)
    
    model.eval()
    
    # 创建示例输入用于图追踪
    example_inputs = (torch.randn(1, 3, 32, 32),)
    
    # 设置量化配置
    qconfig_dict = {"": get_default_qconfig(backend)}
    
    # 准备模型
    model_prepared = prepare_fx(model, qconfig_dict, example_inputs)
    print("FX模式:模型准备完成")
    
    # 校准
    print("开始FX校准...")
    calibrate_model(model_prepared, calibration_loader, num_batches=10)  # 明确指定批次数
    print("FX校准完成")
    
    # 转换
    model_quantized = convert_fx(model_prepared)
    print("FX模式:模型量化完成")
    
    return model_quantized

def quantize_and_evaluate(model, quantize_func, testloader, calibration_loader=None, backend='x86'):
    """
    对模型进行量化并评估其性能。
    """
    quantized_model = quantize_func(model, calibration_loader, backend) if calibration_loader else quantize_func(model)

    # 评估量化模型
    timed(evaluate_model)(quantized_model, testloader)
    
    save_model(quantized_model, f"{model.__class__.__name__.lower()}_quantized.pth")

    compare_model_sizes(model, quantized_model)

if __name__ == '__main__':
    # 获取测试数据
    _, testloader = get_dataloader()

    # 1. 动态量化示例
    print("\n1. 动态量化:")
    float_model = load_model()
    quantize_and_evaluate(float_model, dynamic_quantize_model, testloader)
    
    # 2. 静态量化示例（需要校准数据）
    print("\n2. 静态量化:")
    # 尝试加载可量化模型
    quantizable_model = load_quantizable_model()
    
    # 创建真实的校准数据（从CIFAR-10训练集，无数据增强）
    print("创建校准数据...")
    calibration_dataset = get_calibration_dataset()
    
    # 创建校准数据加载器（使用训练数据的子集）
    calibration_loader = create_calibration_loader(
        calibration_dataset, 
        batch_size=32, 
        num_samples=1000  # 从训练集的50,000个样本中选择1000个进行校准
    )
    
    # eager mode
    print("eager mode:")
    quantize_and_evaluate(quantizable_model, static_quantize_model_eager, testloader, calibration_loader)
    
    # fx graph mode
    print("fx graph mode:")
    quantize_and_evaluate(quantizable_model, static_quantize_model_fx, testloader, calibration_loader)
    
    print("\n=== 量化完成 ===")
