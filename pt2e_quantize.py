'''
PT2E Quantization

1. Export the model to ExportedProgram
2. Annotate the model using a quantizer
3. Prepare the model for quantization
4. Calibrate the model with calibration data
5. Convert the model to a quantized version
6. (Optional) Compile the quantized model for optimized execution
'''
import copy
import torch
import torchvision.models as models
import torchao.quantization.pt2e as pt2e
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)

from quantize import calibrate_model
from utils import load_model, save_model, compare_model_sizes, load_quantizable_model, timed
from dataloader import create_calibration_loader, get_calibration_dataset, get_dataloader
from evaluate import evaluate_model

def pt2e_quantize_model(model, calibration_loader):
    """
    Use PyTorch 2 Export to quantize the model.
    Args:
        model: The model to be quantized (QuantizableResNet)
        calibration_loader: Calibration data loader
        backend: Quantization backend ('x86' or 'qnnpack')
    Returns:
        quantized_model: The quantized model
    """

    # 使用标准内存格式而不是channels_last，避免FX图转换问题
    example_inputs = (torch.randn(1, 3, 32, 32),)
    # example_inputs = (torch.randn(50, 3, 32, 32).contiguous(memory_format=torch.channels_last),)

    # 1. Export the model to ExportedProgram
    with torch.no_grad():
        exported_model = export(model, example_inputs).module()
    
    # 2. Annotate the model using a quantizer
    quantizer = X86InductorQuantizer()
    quantizer.set_global(get_default_x86_inductor_quantization_config())

    # 3. Prepare the model for quantization
    prepared_model = prepare_pt2e(exported_model, quantizer)

    # 4. Calibrate the model with calibration data
    calibrate_model(prepared_model, calibration_loader)

    # 5. Convert the model to a quantized version
    quantized_model = convert_pt2e(prepared_model)

    # 6. (Optional) Compile the quantized model for optimized execution
    torch.compile(quantized_model)
    
    # 确保返回正确的模型
    return quantized_model

def pt2e_quantize_and_evaluate(model, testloader, calibration_loader=None):
    """ Use PyTorch 2 Export to quantize the model and evaluate its performance.

    Args:
        model: The model to be quantized (QuantizableResNet)
        testloader: The test data loader
        calibration_loader: The calibration data loader
        backend: Quantization backend ('x86' or 'qnnpack')
    Returns:
        quantized_model: The quantized model
    """
    quantized_model = pt2e_quantize_model(model, calibration_loader)

    timed(evaluate_model)(quantized_model, testloader)

    # save_model(quantized_model, f'pt2e_quantized_model.pth')

    # compare_model_sizes(model, quantized_model)

if __name__ == '__main__':
    # Get data loaders
    _, testloader = get_dataloader()

    # model_name = "resnet18"
    # model = models.__dict__[model_name](pretrained=True)
    model = load_model().to('cpu')
    # quantizable_model = load_quantizable_model()
    # model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()

    calibration_dataset = get_calibration_dataset()

    calibration_loader = create_calibration_loader(
        calibration_dataset, 
        batch_size=32, 
        num_samples=1000  # Use 1000 samples for calibration
    )

    pt2e_quantize_and_evaluate(model, testloader, calibration_loader)