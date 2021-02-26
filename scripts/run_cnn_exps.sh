#!/bin/sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pickle-stats
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pickle-stats
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pickle-stats
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pickle-stats
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pickle-stats
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pickle-stats
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pickle-stats
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pickle-stats
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pickle-stats
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pickle-stats

CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pseudo-init --pickle-stats --seed 0
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pseudo-init --pickle-stats --seed 1
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pseudo-init --pickle-stats --seed 2
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pseudo-init --pickle-stats --seed 3
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pseudo-init --pickle-stats --seed 4
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pseudo-init --pickle-stats --seed 5
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pseudo-init --pickle-stats --seed 6
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pseudo-init --pickle-stats --seed 7
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pseudo-init --pickle-stats --seed 8
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --pseudo-init --pickle-stats --seed 9

python eval_cnn.py
