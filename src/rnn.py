import torch
import numpy as np
from torch.nn.parameter import Parameter
from network_init import get_quantum_uniform
from network_init import pseudo_quantum_uniform


def generate_data_adding(time_steps, n_data):
    """
    Generate data for the adding problem.
    Source:
        https://github.com/amarshah/complex_RNN/blob/master/adding_problem.py
    Params:
        time_steps: The number of time steps we would like to consider.
        n_data: the number of sequences we would like to consider.
    returns:
        x: [n_data, time_steps, 2] input array.
        y: [n_data, 1] output array.
    """
    x = np.asarray(np.zeros((time_steps, int(n_data), 2)),
                   dtype=np.float)
    # this should be low=-1!? According to hochreiter et al?!
    x[:, :, 0] = np.asarray(np.random.uniform(low=0.,
                                              high=1.,
                                              size=(time_steps, n_data)),
                            dtype=np.float)
    inds = np.asarray(np.random.randint(time_steps/2, size=(n_data, 2)))
    inds[:, 1] += int(time_steps/2)

    for i in range(int(n_data)):
        x[inds[i, 0], i, 1] = 1.0
        x[inds[i, 1], i, 1] = 1.0

    y = (x[:, :, 0] * x[:, :, 1]).sum(axis=0)
    y = np.reshape(y, (n_data, 1))
    return x.transpose([1, 0, 2]), y


def generate_data_memory(time_steps, n_data, n_sequence=10):
    """
    Generate data for the memory problem.
    Source:
        https://github.com/amarshah/complex_RNN/blob/master/memory_problem.py
    Params:
        time_steps: The number of time steps we would like to consider.
        n_data: the number of sequences we would like to consider.
        n_sequence: The length of the initial sequence to be memorized.
                    This number is added to the total length.
    returns:
        x: [n_data, time_steps + n_sequence*2] input array.
        y: [n_data, time_steps + n_sequence*2] output array.
    """

    seq = np.random.randint(1, high=9, size=(n_data, n_sequence))
    zeros1 = np.zeros((n_data, time_steps - 1))
    zeros2 = np.zeros((n_data, time_steps))
    marker = 9 * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))

    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int32')
    return x, y


class LSTMCell(torch.nn.Module):
    def __init__(self, hidden_size=250,
                 input_size=1, output_size=1,
                 randomness_file=None):
        """Create a Long Short Term Memory cell as described at
           https://arxiv.org/pdf/1503.04069.pdf
        Args:
            hidden_size (int, optional): The cell size. Defaults to 250.
            input_size (int, optional): The number of input dimensions.
                                        Defaults to 1.
            output_size (int, optional): Output dimension number.
                                         Defaults to 1.
        """
        super().__init__()
        self.randomness_file = randomness_file
        self.hidden_size = hidden_size
        self.output_size = output_size
        # create the weights
        self.Wz = Parameter(torch.Tensor(input_size, hidden_size))
        self.Wi = Parameter(torch.Tensor(input_size, hidden_size))
        self.Wf = Parameter(torch.Tensor(input_size, hidden_size))
        self.Wo = Parameter(torch.Tensor(input_size, hidden_size))

        self.pi = Parameter(torch.Tensor(hidden_size))
        self.pf = Parameter(torch.Tensor(hidden_size))
        self.po = Parameter(torch.Tensor(hidden_size))

        self.bz = Parameter(torch.Tensor(hidden_size))
        self.bi = Parameter(torch.Tensor(hidden_size))
        self.bf = Parameter(torch.Tensor(hidden_size))
        self.bo = Parameter(torch.Tensor(hidden_size))

        self.Rz = Parameter(torch.Tensor(output_size, hidden_size))
        self.Ri = Parameter(torch.Tensor(output_size, hidden_size))
        self.Rf = Parameter(torch.Tensor(output_size, hidden_size))
        self.Ro = Parameter(torch.Tensor(output_size, hidden_size))

        self.g = torch.nn.Tanh()
        self.gate_act = torch.nn.Sigmoid()
        self.h = torch.nn.Tanh()

        self.proj = Parameter(torch.Tensor(hidden_size, output_size))

    def reset_parameters(self, init: str,
                         address='tcp://localhost:5555') -> None:
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            if init == 'quantum':
                quantum_random = get_quantum_uniform(weight.shape, -stdv, stdv,
                                                     file=self.randomness_file)
                with torch.no_grad():
                    weight.data.copy_(torch.from_numpy(
                        quantum_random.astype(np.float16)))
            elif init == 'pseudo':
                with torch.no_grad():
                    weight.uniform_(-stdv, stdv)
            elif init == 'pseudoquantum':
                with torch.no_grad():
                    weight.data.copy_(pseudo_quantum_uniform(
                        -stdv, stdv, size=tuple(weight.shape)
                        ))
            else:
                raise ValueError(f'Unknown model "{init}", options are: "quantum",\
                                "pseudo", "pseudoquantum"')

    def forward(self, x, c, ym1) -> tuple:
        z = torch.matmul(x, self.Wz) + torch.matmul(ym1, self.Rz) + self.bz
        z = self.g(z)
        i = torch.matmul(x, self.Wi) + torch.matmul(ym1, self.Ri) \
            + self.pi*c + self.bi
        i = self.gate_act(i)
        f = torch.matmul(x, self.Wf) + torch.matmul(ym1, self.Rf) \
            + self.pf*c + self.bf
        f = self.gate_act(f)
        c = z*i + c*f
        o = torch.matmul(x, self.Wo) + torch.matmul(ym1, self.Ro) \
            + self.po*c + self.bo
        o = self.gate_act(o)
        y = self.h(c)*o
        y = torch.matmul(y, self.proj)
        return (c, y)

    def zero_state(self, batch_size: int) -> tuple:
        return (torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, self.output_size))


if __name__ == '__main__':
    x, y = generate_data_memory(100, 50, 10)
    print('test')
