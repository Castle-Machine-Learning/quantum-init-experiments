#!/bin/sh
echo "Running CNN experiments"
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init quantum --pickle-stats --seed 0 
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init quantum --pickle-stats --seed 1
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init quantum --pickle-stats --seed 2
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init quantum --pickle-stats --seed 3
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init quantum --pickle-stats --seed 4
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init quantum --pickle-stats --seed 5
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init quantum --pickle-stats --seed 6
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init quantum --pickle-stats --seed 7
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init quantum --pickle-stats --seed 8
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init quantum --pickle-stats --seed 9
echo "quantum done"

CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudo --pickle-stats --seed 0
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudo --pickle-stats --seed 1
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudo --pickle-stats --seed 2
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudo --pickle-stats --seed 3
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudo --pickle-stats --seed 4
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudo --pickle-stats --seed 5
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudo --pickle-stats --seed 6
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudo --pickle-stats --seed 7
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudo --pickle-stats --seed 8
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudo --pickle-stats --seed 9
echo "pseudo-init done"


CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudoquantum --pickle-stats --seed 0
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudoquantum --pickle-stats --seed 1
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudoquantum --pickle-stats --seed 2
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudoquantum --pickle-stats --seed 3
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudoquantum --pickle-stats --seed 4
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudoquantum --pickle-stats --seed 5
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudoquantum --pickle-stats --seed 6
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudoquantum --pickle-stats --seed 7
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudoquantum --pickle-stats --seed 8
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudoquantum --pickle-stats --seed 9
echo "pseudoquantum-init done"


python eval_cnn.py
