import pickle
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

print('starting evaluation.')
res = pickle.load(open("stats.pickle", "rb"))

pseudo_acc = []
pseudo_loss = []
quantum_acc = []
quantum_loss = []
pseudoquantum_acc = []
pseudoquantum_loss = []

for exp in res:
    if exp['args'].init == 'pseudo':
        pseudo_acc.append(exp['test_acc_lst'])
        pseudo_loss.append(exp['test_loss_lst'])
    elif exp['args'].init == 'quantum':
        quantum_acc.append(exp['test_acc_lst'])
        quantum_loss.append(exp['test_loss_lst'])
    elif exp['args'].init == 'pseudoquantum':
        pseudoquantum_acc.append(exp['test_acc_lst'])
        pseudoquantum_loss.append(exp['test_loss_lst'])

pseudo_acc = np.array(pseudo_acc)
pseudo_loss = np.array(pseudo_loss)

quantum_acc = np.array(quantum_acc)
quantum_loss = np.array(quantum_loss)

pseudoquantum_acc = np.array(pseudoquantum_acc)
pseudoquantum_loss = np.array(pseudoquantum_loss)


pseudo_acc_mean = np.mean(pseudo_acc, axis=0)
pseudo_acc_std = np.std(pseudo_acc, axis=0)

quantum_acc_mean = np.mean(quantum_acc, axis=0)
quantum_acc_std = np.std(quantum_acc, axis=0)

pseudoquantum_acc_mean = np.mean(pseudoquantum_acc, axis=0)
pseudoquantum_acc_std = np.std(pseudoquantum_acc, axis=0)

x = np.array(range(len(pseudo_acc_mean)))
plt.errorbar(x, pseudo_acc_mean, yerr=pseudo_acc_std, label='pseudo',
             capsize=2)
x = np.array(range(len(quantum_acc_mean)))
plt.errorbar(x, quantum_acc_mean, yerr=quantum_acc_std, label='quantum',
             capsize=2)
x = np.array(range(len(pseudoquantum_acc_mean)))
plt.errorbar(x, pseudoquantum_acc_mean, yerr=pseudoquantum_acc_std,
             label='pseudoquantum', capsize=2)

plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend()
# plt.savefig('random_eval.png')
# tikzplotlib.save('random_eval.tex', standalone=True)
plt.show()
print('plots saved.')
