#!/bin/sh
echo "Running CNN experiments"

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 
do
   echo "Quantum-init experiment no: $i "
   CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init quantum --pickle-stats --seed $i 
done

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 
do
   echo "pseudorandom-init experiment no: $i "
   CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudo --pickle-stats --seed $i 
done


for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 
do
   echo "pseudoquantum-init experiment no: $i "
   CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_cnn.py --init pseudoquantum --pickle-stats --seed $i
done

python eval_cnn.py
