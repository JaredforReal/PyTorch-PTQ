import torch
from quantizable_resnet import create_quantizable_resnet
from resnet import create_resnet
from torch.ao.quantization import get_default_qconfig, prepare, convert
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.utils import data
import os

def timed(func):
    """装饰器：计算函数执行时间"""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken by {func.__name__}: {end - start:.2f} seconds")
        return result
    return wrapper

def load_model(path="./weights/resnet18_cifar10.pth"):
    """加载预训练的模型"""
    model = create_resnet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def load_quantizable_model(path="./weights/resnet18_cifar10.pth"):
    """加载预训练的可量化模型"""
    model = create_quantizable_resnet()
    # Load state dict with strict=False to handle potential key mismatches
    state_dict = torch.load(path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def load_full_model(path):
    """
    加载完整模型（包括量化配置和状态字典）
    """
    full_model = torch.load(path)
    # 不在这里调用eval()，让调用者决定
    return full_model

def save_model(model, name="resnet18_cifar10_quantized.pth"):
    """保存模型的权重和整体"""
    torch.save(model.state_dict(), "./weights/" + name)
    torch.save(model, "./weights/full_" + name)
    print(f"模型权重已保存至: ./weights/{name}")
    print(f"完整模型已保存至: ./weights/full_{name}")

def compare_model_sizes(original_model, quantized_model):
    """
    比较原始模型和量化模型的大小
    """
    def get_model_size(model):
        try:
            # 尝试保存整个模型
            torch.save(model, 'temp_model.pth')
            size = os.path.getsize('temp_model.pth')
            os.remove('temp_model.pth')
            return size
        except Exception as e:
            # 如果无法保存整个模型（如FX量化模型），则只保存state_dict
            print(f"无法保存整个模型 ({e})，使用state_dict计算大小...")
            torch.save(model.state_dict(), 'temp_model.pth')
            size = os.path.getsize('temp_model.pth')
            os.remove('temp_model.pth')
            return size
    
    original_size = get_model_size(original_model)
    print(f"Original model size: {original_size / (1024*1024):.2f} MB")

    quantized_size = get_model_size(quantized_model)
    print(f"Quantized model size: {quantized_size / (1024*1024):.2f} MB")

    print(f"Compression ratio: {original_size / quantized_size:.2f}x")