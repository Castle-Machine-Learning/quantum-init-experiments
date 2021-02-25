#!/bin/sh
python mnist_cnn.py --pickle-stats
python mnist_cnn.py --pickle-stats
python mnist_cnn.py --pickle-stats
python mnist_cnn.py --pickle-stats
python mnist_cnn.py --pickle-stats
python mnist_cnn.py --pickle-stats
python mnist_cnn.py --pickle-stats
python mnist_cnn.py --pickle-stats
python mnist_cnn.py --pickle-stats
python mnist_cnn.py --pickle-stats

python mnist_cnn.py --init=pseudo --pickle-stats --seed 0
python mnist_cnn.py --init=pseudo --pickle-stats --seed 1
python mnist_cnn.py --init=pseudo --pickle-stats --seed 2
python mnist_cnn.py --init=pseudo --pickle-stats --seed 3
python mnist_cnn.py --init=pseudo --pickle-stats --seed 4
python mnist_cnn.py --init=pseudo --pickle-stats --seed 5
python mnist_cnn.py --init=pseudo --pickle-stats --seed 6
python mnist_cnn.py --init=pseudo --pickle-stats --seed 7
python mnist_cnn.py --init=pseudo --pickle-stats --seed 8
python mnist_cnn.py --init=pseudo --pickle-stats --seed 9

python eval_cnn.py
