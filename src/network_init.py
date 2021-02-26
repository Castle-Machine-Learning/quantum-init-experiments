import math
import torch
import numpy as np
from src import FCMLQ


def get_quantum_uniform(shape: tuple, low: float, high: float,
                        address='tcp://localhost:5555') -> np.array:
    """ Get a numpy array with quantum uniformly initialized numbers

    Args:
        shape (ouple): Desired output array shape
        low (float): The lower bound of the uniform distribution
        high (float): The upper bound of the uniform distribution
        address (optional): Location of the quantum randomness
            provider. Defaults to tcp://localhost:5555 .

    Returns:
        uniform (nparray): Array initialized to U(a, b).
    """
    number_count = np.prod(shape)
    zero_one = FCMLQ.request_rnd(number_count, address=address)
    zero_one = np.reshape(zero_one, shape)
    uniform = (high - low) * zero_one + low
    return uniform


def _calculate_fan_in_and_fan_out(tensor):
    # from torch.nn
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for \
            tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def pseudo_quantum_uniform(from_: float, to_:float, size: tuple=1,
                           mean_qubit_value: float=0.475,
                           bits_per_float: int=16):
    # make sure size is a tuple
    if isinstance(size, int):
        size = (size,)

    # this creates a potentially very large intermediate array to store all bits..
    # a different solution might be to build up the values incrementally, sacrificing
    # some performance.
    bits = np.random.binomial(1, p=mean_qubit_value, size=(*size, bits_per_float))
    e = 2.0 ** np.arange(-bits_per_float, 0)
    return (to_-from_)*np.dot(bits, e)+from_


def kaiming_uniform_(tensor: torch.tensor,
                     a=0, fan=None,
                     nonlinearity: str = 'relu',
                     mode='quantum', address='tcp://localhost:5555',
                     ) -> None:
    """ In place-initializtion with a quantum kaiming_uniform initialization.
        The implementation follows:
        https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_

    Args:
        tensor (torch.tensor): The tensor we want to initialize with
            quantum random numbers.
        a (int, optional): the negative slope of the rectifier used after
            this layer (only used with 'leaky_relu'). Defaults to 0.
        fan ([type], optional): [description]. Defaults to None.
        nonlinearity (str, optional): [description]. Defaults to 'relu'.
        quantum (bool, optional): Use pseudorandom numbers if False.
            Defaults to True.
        address (str, optional): Location of the quantum randomness server.
            Defaults to 'tcp://localhost:5555'.
    """
    if not fan:
        fan, _ = _calculate_fan_in_and_fan_out(tensor)

    gain = torch.nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    if mode == 'quantum':
        quantum_random = get_quantum_uniform(tensor.shape, -bound, bound, address=address)
        with torch.no_grad():
            tensor.data.copy_(torch.from_numpy(quantum_random.astype(np.float16)))

    elif mode == 'pseudo':
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    
    elif mode == 'pseudoquantum':
        with torch.no_grad():
            tensor.data.copy_(torch.from_numpy(
                pseudo_quantum_uniform(-bound, bound, size=tuple(tensor.shape)).astype(np.float16)))
   
    else:
        raise ValueError(f'Unknown model "{mode}", options are: "quantum", "pseudo", "pseudoquantum"') 
