import torch

from src.models.baseline import MelanomaBaselineCNN
from src.models.custom import CustomResNet
from src.models.efficientnet import EfficientNetWrapper
from src.models.resnet import ResNet50Wrapper
from src.models.convnext import ConvNeXtV2Wrapper
from src.models.effnet_v2 import EfficientNetV2Wrapper

MODEL_REGISTRY = {
    "baseline_cnn": MelanomaBaselineCNN,
    "custom_resnet": CustomResNet,
    "efficientnet_b0": EfficientNetWrapper,
    "resnet50": ResNet50Wrapper,
    "convnextv2_tiny": ConvNeXtV2Wrapper,
    "tf_efficientnetv2_s": EfficientNetV2Wrapper,
}


def build_model_from_config(model_cfg: dict):
    name = model_cfg.get("name")

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[name]
    return model_cls(model_cfg)


def load_model_weights(model, model_cfg: dict):
    """
    Loads weights into the model from a .pt or .ckpt file.
    Assumes the model architecture matches the weights.
    """

    weights_path = model_cfg.get("weights_path")
    if not weights_path:
        return model

    if weights_path.endswith(".ckpt"):
        ckpt = torch.load(weights_path, map_location="cpu")

        state_dict = ckpt.get("state_dict", ckpt)

        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                clean_state_dict[k.replace("model.", "", 1)] = v
            else:
                clean_state_dict[k] = v

        model.load_state_dict(clean_state_dict, strict=False)
        print(f"[INFO] Loaded Lightning checkpoint weights from {weights_path}")

    elif weights_path.endswith(".pt"):
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded PyTorch weights from {weights_path}")

    else:
        raise ValueError(f"Unsupported weights format: {weights_path}")

    return model
