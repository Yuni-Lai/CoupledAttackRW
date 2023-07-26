#!/usr/bin/env bash
#conda activate py37
#cd ./Attack_RandomWalk_Proximity/02_attack_feature
#nohup bash run.sh > ./run.log 2>&1 &

dataset='KDD99' #['KDD99', 'Musk','Mnist']
for repeat in 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030 #
do
  echo "The current repeat is: $repeat"
#  nohup python -u main.py -dataset $dataset -attack_mode "graph_guided" -attack_loss 'target_anomaly' -random_seed ${repeat} -gpuID 0 > ./results_${dataset}/run1_${repeat}.log 2>&1 &
#  nohup python -u main.py -dataset $dataset -attack_mode "graph_guided" -attack_loss 'attacked_graph' -random_seed ${repeat} -gpuID 0 > ./results_${dataset}/run2_${repeat}.log 2>&1 &
  nohup python -u main.py -dataset $dataset -attack_mode "graph_guided_cf" -attack_loss 'target_anomaly' -random_seed ${repeat} -gpuID 0 > ./results_${dataset}/run3_${repeat}.log 2>&1 &
#  nohup python -u main.py -dataset $dataset -attack_mode "random" -attack_loss 'target_anomaly' -random_seed ${repeat} -gpuID 0 > ./results_${dataset}/run3_${repeat}.log 2>&1 &

  ###apply_constraint:
  #nohup python -u main.py -apply_constraint -dataset "KDD99" -attack_mode "graph_guided" -attack_loss 'target_anomaly' -random_seed ${repeat} -gpuID 0 > ./results_KDD99/run_${repeat}.log 2>&1 &
  #nohup python -u main.py -apply_constraint -dataset "KDD99" -attack_mode "graph_guided" -attack_loss 'attacked_graph' -random_seed ${repeat} -gpuID 0 > ./results_KDD99/run_${repeat}.log 2>&1 &
  #nohup python -u main.py -apply_constraint -dataset "KDD99" -attack_mode "random" -attack_loss 'target_anomaly' -random_seed ${repeat} -gpuID 0 > ./results_KDD99/run_${repeat}.log 2>&1 &
  wait
done

## grid search###
#for lr in 1.0 10.0 #0.001 0.01 0.1 1.0
#do
#  for attack_epoch in 400 500 #100 200 300
#  do
#    echo "The current parameters(lr,epoch) is: $lr,$attack_epoch"
#    nohup python -u main.py -dataset "Musk" -attack_mode "random"  -attack_loss 'target_anomaly' -lr $lr -attack_epoch $attack_epoch -random_seed 2021 -gpuID 0 > ./results_Musk/run_2021.log 2>&1 &
#    nohup python -u main.py -dataset "Musk" -attack_mode "graph_guided"  -attack_loss 'target_anomaly' -lr $lr -attack_epoch $attack_epoch -random_seed 2021 -gpuID 0 > ./results_Musk/run_2021.log 2>&1 &
#    wait
#  done
#done

echo "Proccess Finished!"

