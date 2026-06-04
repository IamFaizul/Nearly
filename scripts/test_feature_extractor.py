import torch
from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights

weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

model.classifier = torch.nn.Identity()

print(model)