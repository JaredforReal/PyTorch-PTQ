"""
train.py

负责模型训练。
"""
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from resnet import create_resnet
from dataloader import get_dataloader
from utils import save_model

def train_model(model, trainloader, num_epochs=10):
    """训练模型"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader))
        for i, data in progress_bar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_description(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / (i + 1):.4f}')
        
        scheduler.step()

    print('Finished Training')
    return model

if __name__ == '__main__':
    # 创建模型
    resnet_model = create_resnet()
    
    # 获取数据
    trainloader, _ = get_dataloader()
    
    # 训练模型
    trained_model = train_model(resnet_model, trainloader, num_epochs=20) # 为了演示，只训练10个epoch
    
    # 保存模型
    save_model(trained_model)
    