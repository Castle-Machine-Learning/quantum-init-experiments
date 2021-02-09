#!/bin/sh

python mnist.py --pseudo-init --pickle-stats --seed 0
python mnist.py --pseudo-init --pickle-stats --seed 1
python mnist.py --pseudo-init --pickle-stats --seed 2

python mnist.py --pickle-stats --qbits 5
python mnist.py --pickle-stats --qbits 5
python mnist.py --pickle-stats --qbits 5

python eval.py