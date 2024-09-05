import torch
import torch.onnx
import segmentation_models_pytorch as smp

size = 256

# model = smp.UnetPlusPlus('timm-resnest14d', encoder_weights=None, classes=1, activation='sigmoid', in_channels=3)
model = smp.UnetPlusPlus('efficientnet-b0', encoder_weights=None, classes=1, activation='sigmoid', in_channels=3)
# model.load_state_dict(torch.load('Best_Dice.pt'))
model.load_state_dict(torch.load('Best_Dice_ver2.pt'))

import torch.nn as nn

for m in model.modules():
    if isinstance(m, nn.Upsample):
        m.recompute_scale_factor = None

model.eval()

dummy_input = torch.randn(1, 3, size, size, device='cpu')

torch.onnx.export(model,
                    dummy_input,
                    "model_ver2.onnx",
                    input_names=["input"],
                    output_names=["output"],
                    verbose=True,
                    opset_version=17)