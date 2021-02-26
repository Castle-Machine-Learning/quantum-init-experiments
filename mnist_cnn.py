# based on https://github.com/pytorch/examples/blob/master/mnist/main.py
# import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pickle
from src.network_init import kaiming_uniform_, _calculate_fan_in_and_fan_out


class Net(nn.Module):
    """ Fully connected parameters have been reduced
        to reduce the number of random numbers required.
    """
    def __init__(self, quantum_init=True,
                 address="tcp://localhost:5555"):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1600, 64)
        self.fc2 = nn.Linear(64, 10)
        self.quantum_init = quantum_init
        if self.quantum_init:
            self.qaddress = address
            # self.qbackend = Aer.get_backend('qasm_simulator')
        else:
            self.qaddress = None

        # initialize weights
        # loop over the parameters
        previous_tensor = None
        for param in self.parameters():
            if param.dim() < 2:
                # use kernel fan for bias terms.
                fan, _ = _calculate_fan_in_and_fan_out(previous_tensor)
            else:
                fan = None

            kaiming_uniform_(param, fan=fan,
                             quantum=self.quantum_init,
                             address=self.qaddress)
            previous_tensor = param

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f},  \
          Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc = 100. * correct / len(test_loader.dataset)
    return test_loss, test_acc


def compute_parameter_total(net):
    total = 0
    for p in net.parameters():
        if p.requires_grad:
            print(p.shape)
            total += np.prod(p.shape)
    return total


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='pseudo-random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before \
                              logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--pseudo-init', action='store_true', default=False,
                        help='If True initialize using real qseudo randomnes')
    parser.add_argument('--pickle-stats', action='store_true', default=False,
                        help='If True stores test loss \
                              and acc in pickle file.')

    args = parser.parse_args()
    print('args', args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.set_deterministic(True)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.pseudo_init:
        print('initializing using pseudorandom numbers.')
        model = Net(quantum_init=False).to(device)
    else:
        print('initializing using quantum randomness.')
        model = Net(quantum_init=True).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    print('weight count:', compute_parameter_total(model))

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    test_loss_lst = []
    test_acc_lst = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        epoch_test_loss, epoch_acc_loss = test(model, device, test_loader)
        test_loss_lst.append(epoch_test_loss)
        test_acc_lst.append(epoch_acc_loss)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    # save plots.
    # plt.plot(test_loss_lst)
    # plt.xlabel('epochs')
    # plt.ylabel('test loss')
    # if args.pseudo_init:
    #     plt.title('pseudo-random-init')
    #     plt.savefig('pseudornd.png')
    # else:
    #     plt.title('quantum-random-init')
    #     plt.savefig('qrnd.png')

    if args.pickle_stats:
        try:
            res = pickle.load(open("stats.pickle", "rb"))
        except (OSError, IOError) as e:
            res = []
            print(e, 'stats.pickle does not exist, \
                  creating a new file.')

        res.append({'args': args,
                    'test_loss_lst': test_loss_lst,
                    'test_acc_lst': test_acc_lst})
        pickle.dump(res, open("stats.pickle", "wb"))
        print('stats.pickle saved.')


if __name__ == '__main__':
    main()
