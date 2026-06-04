import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

model = efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    in_features=1280,
    out_features=13
)

model.load_state_dict(
    torch.load(
        "models/nearly_fashion_model.pth",
        map_location="cpu"
    )
)

print("Fashion model loaded.")

model.classifier = nn.Identity()

print("Classifier removed.")
print(model)