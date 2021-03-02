#!/bin/sh


for i in 0 1 2 3 4 5 6 7 8 9
do
   echo "Quantum-init experiment no: $i "
   CUBLAS_WORKSPACE_CONFIG=:4096:8 python adding_memory_rnn.py --problem adding --init quantum --pickle-stats --seed $i 
done

for i in 0 1 2 3 4 5 6 7 8 9
do
   echo "pseudorandom-init experiment no: $i "
   CUBLAS_WORKSPACE_CONFIG=:4096:8 python adding_memory_rnn.py --problem adding --init pseudo --pickle-stats --seed $i 
done


for i in 0 1 2 3 4 5 6 7 8 9
do
   echo "pseudoquantum-init experiment no: $i "
   CUBLAS_WORKSPACE_CONFIG=:4096:8 python adding_memory_rnn.py --problem adding --init pseudoquantum --pickle-stats --seed $i
done


for i in 0 1 2 3 4 5 6 7 8 9
do
   echo "Quantum-init experiment no: $i "
   CUBLAS_WORKSPACE_CONFIG=:4096:8 python adding_memory_rnn.py --problem memory --init quantum --pickle-stats --seed $i 
done

for i in 0 1 2 3 4 5 6 7 8 9
do
   echo "pseudorandom-init experiment no: $i "
   CUBLAS_WORKSPACE_CONFIG=:4096:8 python adding_memory_rnn.py --problem memory --init pseudo --pickle-stats --seed $i 
done

for i in 0 1 2 3 4 5 6 7 8 9
do
   echo "pseudoquantum-init experiment no: $i "
   CUBLAS_WORKSPACE_CONFIG=:4096:8 python adding_memory_rnn.py --problem memory --init pseudoquantum --pickle-stats --seed $i
done




python adding_memory_rnn.py --pickle-stats --problem adding 
python adding_memory_rnn.py --pickle-stats --problem memory
python adding_memory_rnn.py --pickle-stats --problem adding --pseudo-init
python adding_memory_rnn.py --pickle-stats --problem memory --pseudo-init