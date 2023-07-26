#!/usr/bin/env bash
#conda activate py37
#cd /home/user/MyProjects/Attack_RandomWalk/Attack_RandomWalk_Bipartite/
#cd /home/laiyuni/MyProjects/Attack_RandomWalk/Attack_RandomWalk_Bipartite/
#nohup bash run.sh > ./run.log 2>&1 &
for repeat in 2024 2025 2026 2027 2028 2029 2030
do
  echo "The current repeat is: $repeat"
  #nohup python -u main.py -dataset "Magzine" -random_seed ${repeat} > ./results_Magzine/run_${repeat}.log 2>&1 &
  #nohup python -u main.py -dataset "AuthorPapers" -attack_epoch 40 -random_seed ${repeat} > ./results_AuthorPapers/run_${repeat}.log 2>&1 &
  #  nohup python -u main.py -dataset "AuthorPapers" -attack_mode 'random' -random_seed ${repeat} > ./results_AuthorPapers/run_${repeat}.log 2>&1 &
#  nohup python -u main.py -dataset "AuthorPapers" -attack_mode 'degree' -random_seed ${repeat} > ./results_AuthorPapers/run_${repeat}.log 2>&1 &
#  nohup python -u main.py -dataset "Magzine" -attack_mode 'random' -random_seed ${repeat} > ./results_Magzine/run_${repeat}.log 2>&1 &
#  nohup python -u main.py -dataset "Magzine" -attack_mode 'degree' -random_seed ${repeat} > ./results_Magzine/run_${repeat}.log 2>&1 &
#  nohup python -u main.py -dataset "AuthorPapers" -attack_mode 'DeepWalk' -random_seed ${repeat} > ./results_AuthorPapers/run_${repeat}.log 2>&1 &
  nohup python -u main.py -dataset "Magzine" -attack_mode 'DeepWalk' -random_seed ${repeat} > ./results_Magzine/run_${repeat}.log 2>&1 &
  wait
done


##grid search###
#opt='Adam'
#dataset='AuthorPapers' # Magzine AuthorPapers
##for lr in 0.1 0.5 1.0
##do
#for lamda in 1e-3 1e-4 1e-5 1e-6 1e-7 #1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9 1e-10
#do
#  echo "The current parameters(lamda) is: $lamda"
##  nohup python -u main.py -dataset $dataset -lr 0.01 -opt $opt -lamda $lamda -attack_epoch 50 -random_seed 2030 -gpuID 0 > ./results_${dataset}/run1_2030.log 2>&1 &
##  nohup python -u main.py -dataset $dataset -lr 0.1 -opt $opt -lamda $lamda -attack_epoch 50 -random_seed 2030 -gpuID 0 > ./results_${dataset}/run2_2030.log 2>&1 &
##  nohup python -u main.py -dataset $dataset -lr 1.0 -opt $opt -lamda $lamda -attack_epoch 50 -random_seed 2030 -gpuID 0 > ./results_${dataset}/run3_2030.log 2>&1 &
#  nohup python -u main.py -dataset $dataset -lr 0.001 -opt $opt -scaling -lamda $lamda -attack_epoch 50 -random_seed 2030 -gpuID 0 > ./results_${dataset}/run4_2030.log 2>&1 &
#  nohup python -u main.py -dataset $dataset -lr 0.01 -opt $opt -scaling -lamda $lamda -attack_epoch 50 -random_seed 2030 -gpuID 0 > ./results_${dataset}/run4_2030.log 2>&1 &
#  nohup python -u main.py -dataset $dataset -lr 0.1 -opt $opt -scaling -lamda $lamda -attack_epoch 50 -random_seed 2030 -gpuID 0 > ./results_${dataset}/run5_2030.log 2>&1 &
#  nohup python -u main.py -dataset $dataset -lr 1.0 -opt $opt -scaling -lamda $lamda -attack_epoch 50 -random_seed 2030 -gpuID 0 > ./results_${dataset}/run6_2030.log 2>&1 &
#  wait
#done
##done

