#!/usr/bin/env bash
#conda activate py37
#cd ./Attack_RandomWalk_Proximity/01_attack_graph

#nohup bash run.sh > ./run.log 2>&1 &

for repeat in 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030
do
  echo "The current repeat is: $repeat"
#  nohup python -u main.py -dataset "KDD99" -attack_mode 'closed-form' -random_seed ${repeat} -gpuID 0 > ./results_KDD99/run1_${repeat}.log 2>&1 &
#  nohup python -u main.py -dataset "KDD99" -attack_mode 'alternative' -random_seed ${repeat} -gpuID 0 > ./results_KDD99/run2_${repeat}.log 2>&1 &
  nohup python -u main.py -dataset "KDD99" -attack_mode 'DeepWalk' -random_seed ${repeat} -gpuID 4 > ./results_KDD99/run_${repeat}.log 2>&1 &
#  nohup python -u main.py -dataset "Mnist" -attack_mode 'closed-form' -random_seed ${repeat} -gpuID 0 > ./results_Mnist/run1_${repeat}.log 2>&1 &
#  nohup python -u main.py -dataset "Mnist" -attack_mode 'alternative' -random_seed ${repeat} -gpuID 0 > ./results_Mnist/run2_${repeat}.log 2>&1 &
  nohup python -u main.py -dataset "Mnist" -attack_mode 'DeepWalk' -random_seed ${repeat} -gpuID 4 > ./results_Mnist/run2_${repeat}.log 2>&1 &
  wait
done


## grid search###
#for lr in 0.001 0.01 0.1 #1.0
#do
#  for lamda in 1e-3 1e-4 1e-5 1e-6 #1e-7
#  do
#    echo "The current parameters(lr,lamda) is: $lr,$lamda"
#    nohup python -u main.py -dataset "Mnist" -attack_mode 'closed-form' -lr $lr -lamda $lamda -attack_epoch 200 -random_seed 2021 -gpuID 0 > ./results_Mnist/run1_2021.log 2>&1 &
#    nohup python -u main.py -dataset "Mnist" -attack_mode 'closed-form' -lr $lr -lamda $lamda -attack_epoch 300 -random_seed 2021 -gpuID 0 > ./results_Mnist/run2_2021.log 2>&1 &
#    wait
#  done
#done

#for epoch in 80 90 100 110
#do
#  nohup python -u main.py -dataset "Mnist" -attack_epoch $epoch -attack_mode 'closed-form' -random_seed 2021 -gpuID 0 > ./results_Mnist/run_${epoch}.log 2>&1 &
#done
echo "Proccess Finished!"
