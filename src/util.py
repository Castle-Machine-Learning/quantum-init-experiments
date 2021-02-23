"""
Based on:
https://github.com/v0lta/Wavelet-network-compression/blob/master/util.py
"""
import numpy as np


def compute_parameter_total(net):
    total = 0
    for p in net.parameters():
        if p.requires_grad:
            print(p.shape)
            total += np.prod(p.shape)
    return total


def pd_to_string(pd_var) -> str:
    '''
    Convert a parameter dict to string
    :param pd_var: The Parameter dictionary
    :return: A string containg what was in the dict.
    '''
    pd_var = pd_var.copy()
    pd_var_str = ''
    for key in list(pd_var.keys()):
        if type(pd_var[key]) is str:
            pd_var_str += '_' + key + pd_var[key]
        elif type(pd_var[key]) is bool:
            pd_var_str += '_' + key
        else:
            pd_var_str += '_' + key + str(pd_var[key])
    return pd_var_str
