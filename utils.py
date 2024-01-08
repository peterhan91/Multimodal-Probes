import torch

def adjust_precision(activation_tensor, output_precision=8, per_channel=True, cos_sim=False):
    '''
    Adjust the precision of the activation subset
    '''
    if output_precision == 64:
        return activation_tensor.to(torch.float64)

    elif output_precision == 32:
        return activation_tensor.to(torch.float32)

    elif output_precision == 16:
        return activation_tensor.to(torch.float16)

    elif output_precision == 8 and not per_channel:
        min_val = activation_tensor.min().item() if not cos_sim else -1
        max_val = activation_tensor.max().item() if not cos_sim else 1
        num_quant_levels = 2**output_precision
        scale = (max_val - min_val) / (num_quant_levels - 1)
        zero_point = round(-min_val / scale)
        return torch.quantize_per_tensor(activation_tensor, scale, zero_point, torch.quint8)

    elif output_precision == 8 and per_channel:
        min_vals = activation_tensor.min(dim=0)[0] if not cos_sim else -1
        max_vals = activation_tensor.max(dim=0)[0] if not cos_sim else 1
        num_quant_levels = 2**output_precision
        scale = (max_vals - min_vals) / (num_quant_levels - 1)
        zero_point = torch.round(-min_vals / scale)
        return torch.quantize_per_channel(activation_tensor, scale, zero_point, 1,  torch.quint8)

    else:
        raise ValueError(f'Invalid output precision: {output_precision}')