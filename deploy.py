"""
deploy.py

演示如何加载量化模型并进行推理。
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
from resnet import create_resnet
from quantize import dynamic_quantize_model

# CIFAR-10 类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def load_inference_model(float_model_path="resnet18_cifar10.pth"):
    """加载用于推理的动态量化模型"""
    # 1. 加载原始浮点模型
    base_model = create_resnet()
    base_model.load_state_dict(torch.load(float_model_path))
    base_model.eval()
    
    # 2. 对其进行动态量化
    quantized_model = dynamic_quantize_model(base_model)
    
    # 将模型移至CPU，因为量化主要在CPU上优化
    quantized_model.to(torch.device('cpu'))
    
    print("Dynamic quantized model loaded and ready for inference on CPU.")
    return quantized_model

def predict(model, image_path):
    """对单张图片进行预测"""
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        print("Creating a dummy image for demonstration.")
        # 创建一个随机的RGB图像
        dummy_tensor = torch.rand(3, 224, 224)
        img = transforms.ToPILImage()(dummy_tensor)


    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0) # 创建一个batch
    
    model.eval()
    with torch.no_grad():
        out = model(batch_t)
    
    _, predicted_idx = torch.max(out, 1)
    predicted_class = classes[predicted_idx.item()]
    
    print(f"Image: {image_path}")
    print(f"Predicted class: '{predicted_class}'")
    return predicted_class

if __name__ == '__main__':
    # 加载量化模型
    inference_model = load_inference_model()
    
    # 进行预测
    # 注意：你需要提供一张图片的路径。这里我们使用一个占位符。
    # 你可以下载一张图片并命名为 'sample_image.jpg' 放在项目根目录。
    sample_image = 'sample_image.jpg'
    predict(inference_model, sample_image)

    # 示例：预测另一张图片
    # sample_image_2 = 'another_image.png'
    # predict(inference_model, sample_image_2)
