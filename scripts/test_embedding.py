from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

from torchvision.models import EfficientNet_B0_Weights

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights = weights)
print(model)

model = efficientnet_b0