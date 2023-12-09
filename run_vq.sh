#!/bin/bash

n_heads=(1 2 4 8 16 32)
model_names=("VQ_1_head" "VQ_2_head" "VQ_4_head" "VQ_8_head" "VQ_16_head" "VQ_32_head")
num_task=1
# 计数器，用于选择 CUDA 设备
n_model=0
num_model=${#model_names[@]}

# 创建一个名为 commands 的空数组
declare -a commands

for ((j=0; j<$num_model; j++))
do
    n_head=${n_heads[$j]}
    model_name=${model_names[$j]}
    # 计算 CUDA 设备 ID
    device_id=$((n_model % 3 + 1))
    rl_device="cuda:${device_id}"
    sim_device="cuda:${device_id}"
    nohup python run_vq.py \
    --headless 
    --rl_device $rl_device \
    --sim_device $sim_device \
    --n_head $n_head \
    --use_forward 0 --stop_gradient 0 --codebook_size 64 \
    --model_name ${n_head}_head > output_vq_${n_head}.out &
    # 更新计数器

    n_model=$((n_model + 1))
    
done

