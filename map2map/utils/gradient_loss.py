import torch

def gradient_loss(pred, true):
    """
    pred, true: [B, 3, D, H, W]  disp x,y,z
    """
    losses = []
    for axis in (-3, -2, -1):            # D, H, W
        dp = torch.diff(pred,  n=1, dim=axis)
        dt = torch.diff(true,  n=1, dim=axis)
        losses.append(torch.mean((dp - dt).pow(2)))
    return sum(losses) / len(losses)