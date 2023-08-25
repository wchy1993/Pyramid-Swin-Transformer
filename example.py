import torch
from Pyramid_Swin_Transformer import PyramidSwinTransformer


def test_updated_PyramidSwinTransformer():
    # Creating a model instance using the PyramidSwinTransformer class definition
    model = PyramidSwinTransformer(
        hidden_dim=96,
        layers=(8, 6, 4, 4),
        heads=(3, 6, 12, 24),
        window_sizes=((4, 8, 16, 32), (4, 8, 16), (4, 8), (4, 8)),
        head_dim=32,
        relative_pos_embedding=True
    )

    input_tensor = torch.randn(8, 3, 256, 256)
    output = model(input_tensor)
    assert isinstance(output, torch.Tensor), "Model output is not a tensor"
    return output.shape


