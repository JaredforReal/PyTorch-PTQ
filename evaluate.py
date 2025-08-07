"""
evaluate.py

评估模型性能。
"""
import torch
from dataloader import get_dataloader
from utils import timed, load_model
import torchao.quantization.pt2e as pt2e

def evaluate_model(model, testloader):
    """在测试集上评估模型准确率"""
    device = torch.device("cpu")
    # 注意：深度学习的量化模型通常在CPU上运行
    model.to(device)

    # 安全地设置为评估模式
    pt2e.move_exported_model_to_eval(model)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
    return accuracy

if __name__ == '__main__':
    # 获取测试数据
    _, testloader = get_dataloader()

    # --- 评估原始浮点模型 ---
    print("Evaluating original float model...")
    
    float_model = load_model()  # 加载预训练的浮点模型
    timed(evaluate_model)(float_model, testloader)