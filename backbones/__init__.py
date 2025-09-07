def get_model(name, **kwargs):
    # ----------------------------------------------------------------
    # 1. Standard IResNet Backbones (from iresnet.py)
    # ----------------------------------------------------------------
    if name in ["r18", "r34", "r50", "r100", "r200"]:
        from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
        model_map = {
            "r18": iresnet18,
            "r34": iresnet34,
            "r50": iresnet50,
            "r100": iresnet100,
            "r200": iresnet200,
        }
        return model_map[name](pretrained=False, **kwargs)

    # ----------------------------------------------------------------
    # 2. Custom IResNet with Preprocessing
    # ----------------------------------------------------------------
    # 2.1. Model from custom_iresnet.py (Input: Grayscale + Mask + Noise)
    elif name in ["r50_custom_v1", "r100_custom_v1"]:
        from .custom_iresnet import iresnet50, iresnet100
        model_map = {
            "r50_custom_v1": iresnet50,
            "r100_custom_v1": iresnet100,
        }

        return model_map[name](pretrained=False, **kwargs)

    # 2.2. Model from custom2_iresnet.py (Input: Grayscale + Mask)

    elif name in ["r50_custom_v2", "r100_custom_v2"]:
        from .custom2_iresnet import iresnet50, iresnet100
        model_map = {
            "r50_custom_v2": iresnet50,
            "r100_custom_v2": iresnet100,
        }
        return model_map[name](pretrained=False, **kwargs)

    # ----------------------------------------------------------------
    # 3. IResNet with FAN Layer (from iresnet_fan.py)
    # ----------------------------------------------------------------
    elif name == "r50_fan":
        # Phiên bản KHÔNG có Fourier Feature Mapping (đầu vào 3 kênh)
        from .iresnet_fan import iresnet50
        return iresnet50(pretrained=False, use_ffm=False, use_fan=True, **kwargs)

    elif name == "r50_fan_ffm":
        # Phiên bản CÓ Fourier Feature Mapping (đầu vào 128 kênh)
        # Đây là phiên bản bạn cần để nạp checkpoint.
        from .iresnet_fan import iresnet50
        return iresnet50(pretrained=False, use_ffm=True, use_fan=True, **kwargs)

    # ----------------------------------------------------------------
    # 4. Specialized IResNet Models (from iresnet2060.py)
    # ----------------------------------------------------------------
    elif name == "r2060":
        from .iresnet2060 import iresnet2060
        return iresnet2060(pretrained=False, **kwargs)

    # ----------------------------------------------------------------
    # 5. MobileFaceNet Backbones (from mobilefacenet.py)
    # ----------------------------------------------------------------
    elif name == "mbf":
        from .mobilefacenet import get_mbf
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)

    elif name == "mbf_large":
        from .mobilefacenet import get_mbf_large
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf_large(fp16=fp16, num_features=num_features)

    # ----------------------------------------------------------------
    # 6. Vision Transformer (ViT) Backbones (from vit.py)
    # ----------------------------------------------------------------
    elif name.startswith("vit"):
        from .vit import VisionTransformer
        num_features = kwargs.get("num_features", 512)

        if name == "vit_t":
            return VisionTransformer(
                img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
                num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)
        elif name == "vit_s":
            return VisionTransformer(
                img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
                num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)
        elif name == "vit_b":
            return VisionTransformer(
                img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
                num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, using_checkpoint=True)
        else:
             raise ValueError(f"Model ViT '{name}' không được định nghĩa.")

    # ----------------------------------------------------------------
    # Fallback for undefined models
    # ----------------------------------------------------------------
    else:
        raise ValueError(f"Model '{name}' không được hỗ trợ hoặc chưa được định nghĩa trong __init__.py.")