__all__ = ["quantizer"]


def quantizer(model, dtype=None, layers=None, engine="fbgemm"):
    """Quantize model to speedup inference but reduce accuray

    Parameters
        model: Transformer model to quantize
        dtype: Dtype to apply to selected layers
        layers: layers to quantize.
        engine: quantization engine to use
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "Import Error"
        )

    if dtype is None:
        dtype = torch.qint8

    if layers is None:
        layers = {torch.nn.Linear}

    torch.backends.quantized.engine = engine
    return torch.quantization.quantize_dynamic(model, layers, dtype)