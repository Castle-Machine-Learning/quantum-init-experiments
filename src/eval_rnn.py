import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


def tensorboard_average(y, window):
    '''
    * The smoothing algorithm is a simple moving average, which, given a
     * point p and a window w, replaces p with a simple average of the
     * points in the [p - floor(w/2), p + floor(w/2)] range.
    '''
    if window > 0:
        window_vals = []
        length = y.shape[-1]
        for p_no in range(0, length, window):
            if p_no > window/2 and p_no < length - window/2:
                window_vals.append(np.mean(y[p_no-int(window/2):p_no+int(window/2)]))
        return np.array(window_vals)
    else:
        return y


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Plot resuts from a CNN experiment')
    parser.add_argument('--logfile', type=str,
                        default='./log/rnn_stats_unbiased.pickle')
    eval_args = parser.parse_args()

    print('starting rnn evaluation.')
    res = pickle.load(open(eval_args.logfile, "rb"))

    adding_pseudo_loss = []
    adding_quantum_loss = []
    adding_pseudoquantum_loss = []

    memory_pseudo_loss = []
    memory_quantum_loss = []
    memory_pseudoquantum_loss = []

    for exp in res:
        if exp['args'].problem == 'adding':
            if exp['args'].init == 'pseudo':
                adding_pseudo_loss.append(exp['train_loss_lst'])
            elif exp['args'].init == 'quantum':
                adding_quantum_loss.append(exp['train_loss_lst'])
            elif exp['args'].init == 'pseudoquantum':
                adding_pseudoquantum_loss.append(exp['train_loss_lst'])
        if exp['args'].problem == 'memory':
            if exp['args'].init == 'pseudo':
                memory_pseudo_loss.append(exp['train_loss_lst'])
            elif exp['args'].init == 'quantum':
                memory_quantum_loss.append(exp['train_loss_lst'])
            elif exp['args'].init == 'pseudoquantum':
                memory_pseudoquantum_loss.append(exp['train_loss_lst'])

    adding_pseudo_loss = np.array(adding_pseudo_loss)
    adding_quantum_loss = np.array(adding_quantum_loss)
    adding_pseudoquantum_loss = np.array(adding_pseudoquantum_loss)

    memory_pseudo_loss = np.array(memory_pseudo_loss)
    memory_quantum_loss = np.array(memory_quantum_loss)
    memory_pseudoquantum_loss = np.array(memory_pseudoquantum_loss)

    def mean_std(loss: np.array) -> tuple:
        return tensorboard_average(np.mean(loss, axis=0), 0),\
               tensorboard_average(np.std(loss, axis=0), 0)

    adding_pseudo_mean, adding_pseudo_std = mean_std(adding_pseudo_loss)
    adding_quantum_mean, adding_quantum_std = mean_std(adding_quantum_loss)
    adding_pseudoquantum_mean, adding_pseudoquantum_std = \
        mean_std(adding_pseudoquantum_loss)

    memory_pseudo_mean, memory_pseudo_std = mean_std(memory_pseudo_loss)
    memory_quantum_mean, memory_quantum_std = mean_std(memory_quantum_loss)
    memory_pseudoquantum_mean, memory_pseudoquantum_std = \
        mean_std(memory_pseudoquantum_loss)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def plot(mean: np.array, std: np.array, color: np.array,
             marker: str, label: str) -> None:
        x = np.array(range(len(mean)))
        plt.plot(x, mean, label=label, color=color,
                 marker=marker)
        plt.fill_between(x, mean - std, mean + std,
                         color=color, alpha=0.2)

    plot(adding_pseudo_mean, adding_pseudo_std, colors[0], marker='o',
         label='pseudo')
    plot(adding_quantum_mean, adding_quantum_std, colors[1], marker='s',
         label='quantum')
    plot(adding_pseudoquantum_mean, adding_pseudoquantum_std, colors[2],
         marker='v', label='pseudoquantum')
    plt.ylabel('loss')
    plt.xlabel('updates')
    plt.title('adding problem lstm')
    plt.legend()
    # plt.savefig('random_eval.png')
    tikzplotlib.save('rnn_eval_adding.tex', standalone=True)
    plt.show()
    # memory plot

    window_size = 50

    plot(memory_pseudo_mean, memory_pseudo_std, colors[0], marker='o',
         label='pseudo')
    plot(memory_quantum_mean, memory_quantum_std, colors[1], marker='s',
         label='quantum')
    plot(memory_pseudoquantum_mean, memory_pseudoquantum_std, colors[2],
         marker='v', label='pseudoquantum')
    plt.ylabel('loss')
    plt.xlabel('updates')
    plt.title('memory problem lstm')
    plt.legend()
    tikzplotlib.save('rnn_eval_memory.tex', standalone=True)
    #plt.savefig('rnn_eval.png')
    plt.show()

    print('plots saved.')
