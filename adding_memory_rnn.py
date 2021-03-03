'''
Following
https://github.com/v0lta/Wavelet-network-compression/blob/master/adding_memory_RNN_compression.py
'''

import time
import datetime
import argparse
import torch
import numpy as np
from src.rnn import LSTMCell, generate_data_adding, generate_data_memory
from src.util import pd_to_string, compute_parameter_total
import pickle


def train_test_loop(args, in_x, in_y_gt, iteration_no, cell, loss_fun,
                    train=False, optimizer=None,
                    baseline=None):
    """
    Run the network on the adding or copy memory problems.
    train: if true turns backpropagation on.
    """
    if train:
        optimizer.zero_grad()
        cell.train()
    else:
        cell.eval()

    time_steps = in_x.shape[1]
    c, y = cell.zero_state(batch_size=in_x.shape[0])
    c = c.cuda()
    y = y.cuda()
    # run the RNN
    y_cell_lst = []
    for t in range(time_steps):
        # batch_major format [b,t,d]
        c, y = cell(x=in_x[:, t, :], c=c, ym1=y)
        y_cell_lst.append(y)

    if args.problem == 'memory':
        el = np.prod(in_y_gt[:, -10:].shape).astype(np.float32)
        y_tensor = torch.stack(y_cell_lst, dim=-1)
        loss = loss_fun(y_tensor, in_y_gt)
        mem_res = torch.max(y_tensor[:, :, -10:], dim=1)[1]
        acc_sum = torch.sum(mem_res == in_y_gt[:, -10:]).type(
            torch.float32).detach().cpu().numpy()
        acc = acc_sum/(el*1.0)
    else:
        # only the last output is interesting
        el = in_y_gt.shape[0]
        train_y_gt = in_y_gt.type(torch.float32)
        loss = loss_fun(y, train_y_gt)
        acc_sum = torch.sum(torch.abs(y - train_y_gt) < 0.05).type(
            torch.float32).detach().cpu().numpy()
        acc = acc_sum/(el*1.0)

    cpu_loss = loss.detach().cpu().numpy()

    if train:
        loss.backward()
        # apply gradients
        optimizer.step()
    if iteration_no % 50 == 0:
        print('step', iteration_no, 'loss', cpu_loss, 'baseline:', baseline,
              'acc', acc, 'train', train)
    return cpu_loss, acc_sum


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequence Modeling \
        - Adding and Memory Problems')
    parser.add_argument('--problem', type=str, default='memory',
                        help='choose adding or memory')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Cell size: Default 512.')
    parser.add_argument('--time_steps', type=int, default=64,
                        help='The number of time steps \
                              in the problem, default 64.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='The size of the training batches. default 128')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='The size of the training batches.')
    parser.add_argument('--n_train', type=int, default=int(6e5),
                        help='The size of the training batches. Default 6e5')
    parser.add_argument('--n_test', type=int, default=int(1e4),
                        help='The size of the training batches. Default 1e4')
    parser.add_argument('--init', choices=['quantum', 'pseudo',
                        'pseudoquantum'], default='quantum',
                        help='Set initialization method')
    parser.add_argument('--pickle-stats', action='store_true', default=False,
                        help='If True stores test loss \
                              and acc in pickle file.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='pseudo-random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # Cross entropy not supportet.
    # torch.set_deterministic(True)

    train_iterations = int(args.n_train/args.batch_size)
    test_iterations = int(args.n_test/args.batch_size)
    time_start = time.time()

    print(args)
    pd = vars(args)

    if args.problem == 'memory':
        input_size = 10
        output_size = 10
        x_train, y_train = generate_data_memory(args.time_steps, args.n_train)
        x_test, y_test = generate_data_memory(args.time_steps, args.n_test)
        # --- baseline ----------------------
        baseline = np.log(8) * 10/(args.time_steps + 20)
        print("Baseline is " + str(baseline))
        loss_fun = torch.nn.CrossEntropyLoss()
    elif args.problem == 'adding':
        input_size = 2
        output_size = 1
        x_train, y_train = generate_data_adding(args.time_steps, args.n_train)
        x_test, y_test = generate_data_adding(args.time_steps, args.n_test)
        baseline = 0.167
        loss_fun = torch.nn.MSELoss()
    else:
        raise NotImplementedError()

    # convert x,y into tensors.
    x_train = torch.from_numpy(x_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    x_test = torch.from_numpy(x_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    cell = LSTMCell(input_size=input_size,
                    hidden_size=args.hidden,
                    output_size=output_size)
    cell.cuda()
    cell.reset_parameters(init=args.init)

    pt = compute_parameter_total(cell)
    print('parameter total', pt)
    optimizer = torch.optim.RMSprop(cell.parameters(), args.lr)

    x_train_lst = torch.split(x_train.cuda(), args.batch_size, dim=0)
    y_train_lst = torch.split(y_train.cuda(), args.batch_size, dim=0)
    x_test_lst = torch.split(x_test.cuda(), args.batch_size, dim=0)
    y_test_lst = torch.split(y_test.cuda(), args.batch_size, dim=0)

    train_loss_lst = []
    for train_iteration_no in range(train_iterations):
        x_train_batch = x_train_lst[train_iteration_no]
        y_train_batch = y_train_lst[train_iteration_no]
        if args.problem == 'memory':
            # --- one hot encoding -------------
            x_train_batch = torch.nn.functional.one_hot(
                x_train_batch.type(torch.int64)).type(torch.float32)
            y_train_batch = y_train_batch.type(torch.int64)
        train_loss, _ = train_test_loop(
            args, x_train_batch, y_train_batch, train_iteration_no, cell,
            loss_fun, train=True, optimizer=optimizer, baseline=baseline)
        train_loss_lst.append(train_loss)

    print('training done... testing ...')
    test_loss_lst = []
    test_acc_sum = 0
    test_el_total = 0
    for test_iteration_no in range(test_iterations):
        with torch.no_grad():
            x_test_batch = x_test_lst[test_iteration_no]
            y_test_batch = y_test_lst[test_iteration_no]
            if args.problem == 'memory':
                # --- one hot encoding -------------
                x_test_batch = torch.nn.functional.one_hot(
                    x_test_batch.type(torch.int64)).type(torch.float32)
                y_test_batch = y_test_batch.type(torch.int64)
            test_loss, test_true_sum = train_test_loop(
                args, x_test_batch, y_test_batch, test_iteration_no, cell,
                loss_fun, baseline=baseline)
            test_acc_sum += test_true_sum
            if args.problem == 'memory':
                test_el_total += np.prod(
                    y_test_batch[:, -10:].shape).astype(np.float32)
            else:
                test_el_total += y_test_batch.shape[0]
            test_loss_lst.append(test_loss)
    # assert test_el_total == args.n_test
    print('test_el_total', test_el_total, 'test_acc_sum', test_acc_sum)
    test_acc = test_acc_sum/(test_el_total*1.0)

    print('test loss mean', np.mean(test_loss_lst),
          'test acc', test_acc, 'pt', pt)
    store_lst = [train_loss_lst, test_loss_lst, test_acc, pt]
    pd_str = pd_to_string(pd)
    time_str = str(datetime.datetime.today())
    print('time:', time_str, 'experiment took',
          time.time() - time_start, '[s]')

    stats_file = "./log/rnn_stats.pickle"
    if args.pickle_stats:
        try:
            res = pickle.load(open(stats_file, "rb"))
        except (OSError, IOError) as e:
            res = []
            print(e, 'stats.pickle does not exist, \
                  creating a new file.')

        res.append({'args': args,
                    'train_loss_lst': train_loss_lst,
                    'test_loss_mean': np.mean(test_loss_lst)})
        pickle.dump(res, open(stats_file, "wb"))
        print('stats_rnn.pickle saved.')
