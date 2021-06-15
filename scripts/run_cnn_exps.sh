#!/bin/sh
echo "Running CNN experiments"

for i in 0 1 2 # 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
do
   echo "Quantum-init experiment no: $i "
   CUBLAS_WORKSPACE_CONFIG=:4096:8 python ./src/mnist_cnn.py --init quantum --pickle-stats --seed $i --storage ./numbers/storage-2-manhattan-unshuffled-32bit-080321.pkl --storage-pos $((121930*$i))
done

for i in 0 1 2 # 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
do
   echo "pseudorandom-init experiment no: $i "
   CUBLAS_WORKSPACE_CONFIG=:4096:8 python ./src/mnist_cnn.py --init pseudo --pickle-stats --seed $i 
done

for i in 0 1 2 # 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
do
   echo "pseudoquantum-init experiment no: $i "
   CUBLAS_WORKSPACE_CONFIG=:4096:8 python ./src/mnist_cnn.py --init pseudoquantum --pickle-stats --seed $i
done

python ./src/eval_cnn.py
