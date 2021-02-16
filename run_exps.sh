#!/bin/sh
python mnist.py --pickle-stats
python mnist.py --pickle-stats
python mnist.py --pickle-stats
python mnist.py --pickle-stats
python mnist.py --pickle-stats
python mnist.py --pickle-stats
python mnist.py --pickle-stats
python mnist.py --pickle-stats
python mnist.py --pickle-stats
python mnist.py --pickle-stats

python mnist.py --pseudo-init --pickle-stats --seed 0
python mnist.py --pseudo-init --pickle-stats --seed 1
python mnist.py --pseudo-init --pickle-stats --seed 2
python mnist.py --pseudo-init --pickle-stats --seed 3
python mnist.py --pseudo-init --pickle-stats --seed 4
python mnist.py --pseudo-init --pickle-stats --seed 5
python mnist.py --pseudo-init --pickle-stats --seed 6
python mnist.py --pseudo-init --pickle-stats --seed 7
python mnist.py --pseudo-init --pickle-stats --seed 8
python mnist.py --pseudo-init --pickle-stats --seed 9

python eval.py