from .resnet18 import resnet18
from .preact_resnet18 import preact_resnet18


def build_model(backbone: str, num_classes: int):
    if backbone == "resnet18":
        return resnet18(num_classes=num_classes)
    elif backbone == "preact_resnet18":
        return preact_resnet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
