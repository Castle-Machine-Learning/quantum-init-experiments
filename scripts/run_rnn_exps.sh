#!/bin/sh

python adding_memory_rnn.py --pickle-stats --problem adding 
python adding_memory_rnn.py --pickle-stats --problem memory
python adding_memory_rnn.py --pickle-stats --problem adding --pseudo_init
python adding_memory_rnn.py --pickle-stats --problem memory --pseudo_init