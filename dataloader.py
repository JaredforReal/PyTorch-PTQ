import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloader(batch_size=128):
    """获取CIFAR-10数据加载器"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def calibrate_model(model, data_loader, num_batches=10):
    """
    校准函数：在校准数据上运行模型以收集统计信息
    
    Args:
        model: 准备好的模型（包含观察器）
        data_loader: 校准数据加载器
        num_batches: 用于校准的批次数量
    """
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            model(data)
            if batch_idx >= num_batches - 1:
                break

def create_calibration_loader(dataset, batch_size=32, num_samples=1000):
    """
    创建校准数据加载器
    
    Args:
        dataset: 完整数据集
        batch_size: 批次大小
        num_samples: 校准样本数量
    
    Returns:
        calibration_loader: 校准数据加载器
    """
    # 从完整数据集中采样部分数据用于校准
    indices = torch.randperm(len(dataset))[:num_samples]
    calibration_subset = data.Subset(dataset, indices)
    
    calibration_loader = data.DataLoader(
        calibration_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return calibration_loader

def get_calibration_dataset():
    """
    获取专门用于校准的数据集（训练数据但无数据增强）
    
    Returns:
        calibration_dataset: 用于校准的数据集
    """
    # 使用与测试时相同的变换（无数据增强）
    transform_calibration = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 使用训练集数据但不应用数据增强
    calibration_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_calibration)
    
    return calibration_dataset
    
