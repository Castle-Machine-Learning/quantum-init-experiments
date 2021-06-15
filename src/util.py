"""
Based on: https://github.com/v0lta/Wavelet-network-compression/blob/master/util.py
"""
import numpy as np

def compute_parameter_total(net):
    """Counts all trainable parameters in a network.

    Args:
        net (torch.nn.Module): A module with optimizable parameters.

    Returns:
        [int]: The number of parameters in the network.
    """    
    total = 0
    for p in net.parameters():
        if p.requires_grad:
            print(p.shape)
            total += np.prod(p.shape)
    return total

